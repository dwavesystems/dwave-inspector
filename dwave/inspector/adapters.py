# Copyright 2019 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import uuid
import logging
import itertools
from collections import abc, Counter

import dimod
import dwave.cloud
from dwave.cloud.utils import reformat_qubo_as_ising, uniform_get, active_qubits
from dwave.cloud.events import add_handler
from dwave.cloud.solver import StructuredSolver
from dwave.embedding import embed_bqm
from dwave.embedding.utils import edgelist_to_adjacency
from dwave.system.composites import EmbeddingComposite
from dwave.system.warnings import WarningAction

from dwave.inspector import storage
from dwave.inspector.utils import itemsgetter

__all__ = [
    'from_qmi_response',
    'from_bqm_response',
    'from_bqm_sampleset',
    'from_objects',
    'enable_data_capture',
]

logger = logging.getLogger(__name__)

SUPPORTED_SOLVER_TOPOLOGY_TYPES = {'chimera', 'pegasus', 'zephyr'}

# only one BQM class since dimod 0.10 (with a few aliases/subclasses for default dtype)
BQM_CLASSES = (dimod.BinaryQuadraticModel, )


def enable_data_capture():
    """Enable logging of submitted problems/answers, embedding parameters,
    embedding/sampling warnings, etc. across the Ocean stack.
    """

    def capture_qmi_response(event, obj, args, return_value):
        logger.debug("{!s}(obj={!r}, args={!r}, return_value={!r})".format(
            event, obj, args, return_value))

        topology = _get_solver_topology(obj)
        if topology['type'] not in SUPPORTED_SOLVER_TOPOLOGY_TYPES:
            logger.error('Solver topology {!r} not supported.'.format(topology))
            return

        try:
            storage.add_problem(problem=args, solver=obj, response=return_value)
        except Exception as e:
            logger.error('Failed to store problem: %r', e)

    # subscribe to problems sampled and results returned in the cloud client
    add_handler('after_sample', capture_qmi_response)

    # save all warnings during embedding by default (equivalent to setting
    # `warnings=WarningAction.SAVE` during sampling)
    EmbeddingComposite.warnings_default = WarningAction.SAVE

    # return embedding map/parameters (equivalent to setting
    # `return_embedding=True` during sampling)
    EmbeddingComposite.return_embedding_default = True

    logger.debug("Data capture enabled for embeddings, warnings and QMIs")


def _get_solver_topology(solver, default=None):
    """Safe getter of solver's topology. Older solver definitions do not include
    the ``topology`` property, but we can assume such solvers, if they are
    structured, have a "chimera" topology type.
    """

    try:
        # get topology, verifying structure
        topology = solver.properties['topology']
        _, _ = topology['type'], topology['shape']
        return topology

    except KeyError:
        if hasattr(solver, 'edges') and hasattr(solver, 'nodes'):
            return {'type': 'chimera', 'shape': [16, 16, 4]}

    return default

def solver_data_postprocessed(solver):
    """Returns (possibly an old) solver's updated `data` that includes the
    missing properties like topology. Also, removes large unused properties.
    """

    # make a copy to avoid modifying the original solver.data
    data = copy.deepcopy(solver.data)

    # add missing but used properties
    data['properties'].setdefault('topology', _get_solver_topology(solver))

    # remove unused properties
    del data['properties']['anneal_offset_ranges']

    return data


def _answer_dict(solutions, active_variables, energies, num_occurrences, timing, num_variables):
    return {
        "format": "qp",
        "solutions": [list(map(int, s)) for s in solutions],
        "active_variables": list(map(int, active_variables)),
        "energies": list(map(float, energies)),
        "num_occurrences": list(map(int, num_occurrences)),
        "timing": timing,
        "num_variables": int(num_variables)
    }

def _unembedded_answer_dict(sampleset):
    return {
        "format": "qp",
        "vartype": sampleset.vartype.name,
        "solutions": [list(map(int, s)) for s in sampleset.record.sample],
        "active_variables": list(sampleset.variables),
        "energies": list(map(float, sampleset.record.energy)),
        "num_occurrences": list(map(int, sampleset.record.num_occurrences)),
        "num_variables": len(sampleset.variables)
    }

def _problem_dict(solver_id, problem_type, problem_data, params=None, stats=None):
    return {
        "solver": solver_id,
        "type": problem_type,
        "params": params if params is not None else {},
        "data": _validated_problem_data(problem_data),
        "stats": stats if stats is not None else {}
    }

def _expand_params(solver, params=None, timing=None):
    """Expand a limited set of user-provided params to a full set, substituting
    missing values with solver defaults.
    """
    if params is None:
        params = {}
    if timing is None:
        timing = {}

    default_annealing_time = solver.properties['default_annealing_time']
    default_programming_thermalization = solver.properties['default_programming_thermalization']
    default_readout_thermalization = solver.properties['default_readout_thermalization']

    # figure out `annealing_time`
    if "qpu_anneal_time_per_sample" in timing:
        # actual value is known
        annealing_time = timing["qpu_anneal_time_per_sample"]
    elif "annealing_time" in params:
        annealing_time = params["annealing_time"]
    elif "anneal_schedule" in params:
        anneal_schedule = params["anneal_schedule"]
        annealing_time = anneal_schedule[-1][0]
    else:
        annealing_time = default_annealing_time

    # figure out `anneal_schedule`
    if "anneal_schedule" in params:
        anneal_schedule = params["anneal_schedule"]
    else:
        anneal_schedule = [[0, 0], [annealing_time, 1]]

    flux_biases = params.get('flux_biases')
    initial_state = params.get("initial_state")

    # set each parameter individually because defaults are not necessarily
    # constant; they can depend on one or more other parameters
    # TODO: cast to primary types for safe JSON serialization
    return {
        "anneal_offsets": params.get("anneal_offsets"),
        "anneal_schedule": anneal_schedule,
        "annealing_time": annealing_time,
        "answer_mode": params.get("answer_mode", "histogram"),
        "auto_scale": params.get("auto_scale", True if not flux_biases else False),
        "beta": params.get("beta", 10 if solver.is_vfyc else 1),
        "chains": params.get("chains"),
        "flux_biases": flux_biases,
        "flux_drift_compensation": params.get("flux_drift_compensation", True),
        "h_gain_schedule": params.get("h_gain_schedule", [[0, 1], [annealing_time, 1]]),
        "initial_state": initial_state,
        "max_answers": params.get("max_answers"),
        "num_reads": params.get("num_reads", 1),
        "num_spin_reversal_transforms": params.get("num_spin_reversal_transforms", 0),
        "postprocess": params.get("postprocess", "sampling" if solver.is_vfyc else ""),
        "programming_thermalization": params.get("programming_thermalization", default_programming_thermalization),
        "readout_thermalization": params.get("readout_thermalization", default_readout_thermalization),
        "reduce_intersample_correlation": params.get("reduce_intersample_correlation", False),
        "reinitialize_state": params.get("reinitialize_state", True if initial_state else False)
    }

def _validated_problem_data(data):
    "Basic types validation/conversion."

    try:
        assert data['format'] == 'qp'
        assert isinstance(data['lin'], list)
        assert isinstance(data['quad'], list)

        data['lin'] = [float(v) if v is not None else None for v in data['lin']]
        data['quad'] = list(map(float, data['quad']))
        if 'embedding' in data:
            data['embedding'] = _validated_embedding(data['embedding'])

        return data

    except Exception as e:
        msg = "invalid problem structure and/or data types"
        logger.warning(msg)
        raise ValueError(msg)

def _validated_embedding(emb):
    "Basic types validation/conversion."

    # check structure, casting to `Dict[str, List[int]]` along the way
    try:
        keys = map(str, emb)
        values = [sorted(map(int, chain)) for chain in emb.values()]
        emb = dict(zip(keys, values))

    except Exception as e:
        msg = "invalid embedding structure"
        logger.warning("{}: {}".format(msg, e))
        raise ValueError(msg, e)

    # validate chains are disjoint
    counts = Counter()
    counts.update(itertools.chain(*emb.values()))
    if not all(v == 1 for v in counts.values()):
        msg = "embedding has overlapping chains in target variables: {}".format(
            sorted(k for k, v in counts.items() if v > 1))
        logger.warning(msg)
        raise ValueError(msg)

    return emb

def _problem_stats(response=None, sampleset=None, embedding_context=None):
    "Generate problem stats from available data."

    if embedding_context is None:
        embedding_context = {}

    embedding = embedding_context.get('embedding')
    chain_strength = embedding_context.get('chain_strength')
    chain_break_method = embedding_context.get('chain_break_method')

    # best guess for number of logical/source vars
    if sampleset:
        num_source_variables = len(sampleset.variables)
    elif embedding:
        num_source_variables = len(embedding)
    elif response:
        num_source_variables = len(response.variables)
    else:
        num_source_variables = None

    # best guess for number of target/source vars
    if response:
        target_vars = set(response.variables)
        num_target_variables = len(response.variables)
    elif sampleset and embedding:
        target_vars = {t for s in sampleset.variables for t in embedding[s]}
        num_target_variables = len(target_vars)
    else:
        target_vars = set()
        num_target_variables = None

    # max chain length
    if embedding:
        # consider only active variables in response
        # (so fixed embedding won't falsely increase the max chain len)
        max_chain_length = max(len(target_vars.intersection(chain))
                               for chain in embedding.values())
    else:
        max_chain_length = 1

    return {
        "num_source_variables": num_source_variables,
        "num_target_variables": num_target_variables,
        "max_chain_length": max_chain_length,
        "chain_strength": chain_strength,
        "chain_break_method": chain_break_method,
    }

def _details_dict(response):
    return {
        "id": response.id,
        "label": response.label,
        "status": response.remote_status,
        "solver": response.solver.id,
        "type": response.problem_type,
        "submitted_on": response.time_received.isoformat(),
        "solved_on": response.time_solved.isoformat()
    }

def _warnings(warnings):
    if not warnings:
        return []

    # translate warning classes (given for type) to string names
    data = copy.deepcopy(warnings)
    for warning in data:
        if issubclass(warning['type'], Warning):
            warning.update(type=warning['type'].__name__)
    return data


def from_qmi_response(problem, response, embedding_context=None, warnings=None,
                      params=None, sampleset=None):
    """Construct problem data for visualization based on the low-level sampling
    problem definition and the low-level response.

    Args:
        problem ((list/dict, dict[(int, int), float]) or dict[(int, int), float]):
            Problem in Ising or QUBO form, conforming to solver graph.
            Note: if problem is given as tuple, it is assumed to be in Ising
            variable space, and if given as a dict, Binary variable space is
            assumed. Zero energy offset is always implied.

        response (:class:`dwave.cloud.computation.Future`):
            Sampling response, as returned by the low-level sampling interface
            in the Cloud Client (e.g. :meth:`dwave.cloud.Solver.sample_ising`
            for Ising problems).

        embedding_context (dict, optional):
            A map containing an embedding of logical problem onto the
            solver's graph (the ``embedding`` key) and embedding parameters
            used (e.g. ``chain_strength``).

        warnings (list[dict], optional):
            Optional list of warnings.

        params (dict, optional):
            Sampling parameters used.

        sampleset (:class:`dimod.SampleSet`, optional):
            Optional unembedded sampleset.

    """
    logger.debug("from_qmi_response({!r})".format(
        dict(problem=problem, response=response, response_energies=response['energies'],
             embedding_context=embedding_context, warnings=warnings,
             params=params, sampleset=sampleset)))

    try:
        linear, quadratic = problem
    except:
        linear, quadratic = reformat_qubo_as_ising(problem)

    # make sure lin/quad are not something like dimod views (that handle
    # directed edges, and effectively duplicate biases of symmetric couplings)
    if not isinstance(linear, dict):
        linear = dict(linear)
    if not isinstance(quadratic, dict):
        quadratic = dict(quadratic)

    solver = response.solver
    if not isinstance(response.solver, StructuredSolver):
        raise TypeError("only structured solvers are supported")

    topology = _get_solver_topology(solver)
    if topology['type'] not in SUPPORTED_SOLVER_TOPOLOGY_TYPES:
        raise TypeError("unsupported solver topology type")

    solver_id = solver.id
    problem_type = response.problem_type

    variables = list(response.variables)
    active = active_qubits(linear, quadratic)

    # filter out invalid values (user's error in problem definition), since
    # SAPI ignores them too
    active = {q for q in active if q in solver.variables}

    # sanity check
    active_variables = response['active_variables']
    assert set(active) == set(active_variables)

    solutions = list(map(itemsgetter(*active_variables), response['solutions']))
    energies = response['energies']
    num_occurrences = response.num_occurrences
    num_variables = solver.num_qubits
    timing = response.timing

    # note: we can't use encode_problem_as_qp(solver, linear, quadratic) because
    # visualizer accepts decoded lists (and nulls instead of NaNs)
    problem_data = {
        "format": "qp",         # SAPI non-conforming (nulls vs nans)
        "lin": [uniform_get(linear, v, 0 if v in active else None)
                for v in solver._encoding_qubits],
        "quad": [quadratic.get((q1,q2), 0) + quadratic.get((q2,q1), 0)
                 for (q1,q2) in solver._encoding_couplers
                 if q1 in active and q2 in active]
    }

    # include optional embedding
    if embedding_context is not None and 'embedding' in embedding_context:
        problem_data['embedding'] = embedding_context['embedding']

    # try to reconstruct sampling params
    if params is None:
        params = {'num_reads': int(sum(num_occurrences))}

    # expand with defaults
    params = _expand_params(solver, params, timing)

    # construct problem stats
    problem_stats = _problem_stats(response=response, sampleset=sampleset,
                                   embedding_context=embedding_context)

    data = {
        "ready": True,
        "details": _details_dict(response),
        "data": _problem_dict(solver_id, problem_type, problem_data, params, problem_stats),
        "answer": _answer_dict(solutions, active_variables, energies, num_occurrences, timing, num_variables),
        "warnings": _warnings(warnings),
        "rel": dict(solver=solver),
    }

    if sampleset is not None:
        data["unembedded_answer"] = _unembedded_answer_dict(sampleset)

    logger.trace("from_qmi_response returned %r", data)

    return data


def from_bqm_response(bqm, embedding_context, response, warnings=None,
                      params=None, sampleset=None):
    """Construct problem data for visualization based on the unembedded BQM,
    the embedding used when submitting, and the low-level sampling response.

    Args:
        bqm (:class:`dimod.BinaryQuadraticModel`/:class:`dimod.core.bqm.BQM`):
            Problem in logical (unembedded) space, given as a BQM.

        embedding_context (dict):
            A map containing an embedding of logical problem onto the
            solver's graph (the ``embedding`` key) and embedding parameters
            used (e.g. ``chain_strength``, ``chain_break_method``, etc).

        response (:class:`dwave.cloud.computation.Future`):
            Sampling response, as returned by the low-level sampling interface
            in the Cloud Client (e.g. :meth:`dwave.cloud.solver.sample_ising`
            for Ising problems).

        warnings (list[dict], optional):
            Optional list of warnings.

        params (dict, optional):
            Sampling parameters used.

        sampleset (:class:`dimod.SampleSet`, optional):
            Optional unembedded sampleset.

    """
    logger.debug("from_bqm_response({!r})".format(
        dict(bqm=bqm, response=response, response_energies=response['energies'],
             embedding_context=embedding_context, warnings=warnings,
             params=params, sampleset=sampleset)))

    solver = response.solver
    if not isinstance(response.solver, StructuredSolver):
        raise TypeError("only structured solvers are supported")

    topology = _get_solver_topology(solver)
    if topology['type'] not in SUPPORTED_SOLVER_TOPOLOGY_TYPES:
        raise TypeError("unsupported solver topology type")

    solver_id = solver.id
    problem_type = response.problem_type

    active_variables = response['active_variables']
    active = set(active_variables)

    solutions = list(map(itemsgetter(*active_variables), response['solutions']))
    energies = response['energies']
    num_occurrences = response.num_occurrences
    num_variables = solver.num_qubits
    timing = response.timing

    # bqm vartype must match response vartype
    if problem_type == "ising":
        bqm = bqm.change_vartype(dimod.SPIN, inplace=False)
    else:
        bqm = bqm.change_vartype(dimod.BINARY, inplace=False)

    # get embedding parameters
    if 'embedding' not in embedding_context:
        raise ValueError("embedding not given")
    embedding = embedding_context.get('embedding')
    chain_strength = embedding_context.get('chain_strength')
    chain_break_method = embedding_context.get('chain_break_method')

    # if `embedding` is `dwave.embedding.transforms.EmbeddedStructure`, we don't
    # need `target_adjacency`
    emb_params = dict(embedding=embedding)
    if not hasattr(embedding, 'embed_bqm'):
        # proxy for detecting dict vs. EmbeddedStructure, without actually
        # importing EmbeddedStructure (did not exist in dwave-system<0.9.10)
        target_adjacency = edgelist_to_adjacency(solver.edges)
        emb_params.update(target_adjacency=target_adjacency)

    # get embedded bqm
    bqm_embedded = embed_bqm(bqm,
                             chain_strength=chain_strength,
                             smear_vartype=dimod.SPIN,
                             **emb_params)

    linear, quadratic, offset = bqm_embedded.to_ising()
    problem_data = {
        "format": "qp",         # SAPI non-conforming (nulls vs nans)
        "lin": [uniform_get(linear, v, 0 if v in active else None)
                for v in solver._encoding_qubits],
        "quad": [quadratic.get((q1,q2), 0) + quadratic.get((q2,q1), 0)
                 for (q1,q2) in solver._encoding_couplers
                 if q1 in active and q2 in active],
        "embedding": embedding
    }

    # try to reconstruct sampling params
    if params is None:
        params = {'num_reads': int(sum(num_occurrences))}

    # expand with defaults
    params = _expand_params(solver, params, timing)

    # TODO: if warnings are missing, calculate them here (since we have the
    # low-level response)

    # construct problem stats
    problem_stats = _problem_stats(response=response, sampleset=sampleset,
                                   embedding_context=embedding_context)

    data = {
        "ready": True,
        "details": _details_dict(response),
        "data": _problem_dict(solver_id, problem_type, problem_data, params, problem_stats),
        "answer": _answer_dict(solutions, active_variables, energies, num_occurrences, timing, num_variables),
        "warnings": _warnings(warnings),
        "rel": dict(solver=solver),
    }

    if sampleset is not None:
        data["unembedded_answer"] = _unembedded_answer_dict(sampleset)

    logger.trace("from_bqm_response returned %r", data)

    return data


def from_bqm_sampleset(bqm, sampleset, sampler, embedding_context=None,
                       warnings=None, params=None):
    """Construct problem data for visualization based on the BQM and sampleset
    in logical space (both unembedded).

    In order for the embedded problem/response to be reconstructed, an embedding
    is required in either the sampleset, or as a standalone argument.

    Note:
        This adapter can only provide best-effort estimate of the submitted
        problem and received samples. Namely, because values of logical
        variables in `sampleset` are produced by a chain break resolution
        method, information about individual physical qubit values is lost.

        Please have in mind you will never see "broken chains" when using this
        adapter.

    Args:
        bqm (:class:`dimod.BinaryQuadraticModel`/:class:`dimod.core.bqm.BQM`):
            Problem in logical (unembedded) space, given as a BQM.

        sampleset (:class:`~dimod.sampleset.SampleSet`):
            Sampling response as a sampleset.

        sampler (:class:`~dimod.Sampler` or :class:`~dimod.ComposedSampler`):
            The :class:`~dwave.system.samplers.dwave_sampler.DWaveSampler`-
            derived sampler used to produce the sampleset off the bqm.

        embedding_context (dict, optional):
            A map containing an embedding of logical problem onto the
            solver's graph (the ``embedding`` key) and embedding parameters
            used (e.g. ``chain_strength``). It is optional only if
            ``sampleset.info`` contains it (see `return_embedding` argument of
            :meth:`~dwave.system.composites.embedding.EmbeddingComposite`).

        warnings (list[dict], optional):
            Optional list of warnings.

        params (dict, optional):
            Sampling parameters used.

    """
    logger.debug("from_bqm_sampleset({!r})".format(
        dict(bqm=bqm, sampleset=sampleset, sampler=sampler, warnings=warnings,
             embedding_context=embedding_context, params=params)))

    if not isinstance(sampler, dimod.Sampler):
        raise TypeError("dimod.Sampler instance expected for 'sampler'")

    # get embedding parameters
    if embedding_context is None:
        embedding_context = sampleset.info.get('embedding_context', {})
    if embedding_context is None:
        raise ValueError("embedding_context not given")
    embedding = embedding_context.get('embedding')
    if embedding is None:
        raise ValueError("embedding not given")
    chain_strength = embedding_context.get('chain_strength')

    def find_solver(sampler):
        if hasattr(sampler, 'solver'):
            return sampler.solver

        for child in getattr(sampler, 'children', []):
            try:
                return find_solver(child)
            except:
                pass

        raise TypeError("'sampler' doesn't use DWaveSampler")

    solver = find_solver(sampler)
    if not isinstance(solver, StructuredSolver):
        raise TypeError("only structured solvers are supported")

    topology = _get_solver_topology(solver)
    if topology['type'] not in SUPPORTED_SOLVER_TOPOLOGY_TYPES:
        raise TypeError("unsupported solver topology type")

    solver_id = solver.id
    problem_type = "ising" if sampleset.vartype is dimod.SPIN else "qubo"

    # bqm vartype must match sampleset vartype
    if bqm.vartype is not sampleset.vartype:
        bqm = bqm.change_vartype(sampleset.vartype, inplace=False)

    # if `embedding` is `dwave.embedding.transforms.EmbeddedStructure`, we don't
    # need `target_adjacency`
    emb_params = dict(embedding=embedding)
    if not hasattr(embedding, 'embed_bqm'):
        # proxy for detecting dict vs. EmbeddedStructure, without actually
        # importing EmbeddedStructure (did not exist in dwave-system<0.9.10)
        target_adjacency = edgelist_to_adjacency(solver.edges)
        emb_params.update(target_adjacency=target_adjacency)

    # get embedded bqm
    bqm_embedded = embed_bqm(bqm,
                             chain_strength=chain_strength,
                             smear_vartype=dimod.SPIN,
                             **emb_params)

    # best effort reconstruction of (unembedded/qmi) response/solutions
    # NOTE: we **can not** reconstruct physical qubit values from logical variables
    # (sampleset we have access to has variable values after chain breaks resolved!)
    active_variables = sorted(list(bqm_embedded.variables))
    active_variables_set = set(active_variables)
    logical_variables = list(sampleset.variables)
    var_to_idx = {var: idx for idx, var in enumerate(logical_variables)}
    unembedding = {q: var_to_idx[v] for v, qs in embedding.items() for q in qs}

    # sanity check
    assert set(unembedding) == active_variables_set

    def expand_sample(sample):
        return [int(sample[unembedding[q]]) for q in active_variables]
    solutions = [expand_sample(sample) for sample in sampleset.record.sample]

    # adjust energies to values returned by SAPI (offset embedding)
    energies = list(map(float, sampleset.record.energy - bqm_embedded.offset))

    num_occurrences = list(map(int, sampleset.record.num_occurrences))
    num_variables = solver.num_qubits
    timing = sampleset.info.get('timing')

    linear, quadratic, offset = bqm_embedded.to_ising()
    problem_data = {
        "format": "qp",         # SAPI non-conforming (nulls vs nans)
        "lin": [uniform_get(linear, v, 0 if v in active_variables_set else None)
                for v in solver._encoding_qubits],
        "quad": [quadratic.get((q1,q2), 0) + quadratic.get((q2,q1), 0)
                 for (q1,q2) in solver._encoding_couplers
                 if q1 in active_variables_set and q2 in active_variables_set],
        "embedding": embedding
    }

    # try to get problem id. if not available, auto-generate one
    problem_id = sampleset.info.get('problem_id')
    if problem_id is None:
        problem_id = "local-%s" % uuid.uuid4()

    # try to reconstruct sampling params
    if params is None:
        params = {'num_reads': int(sum(num_occurrences))}

    # expand with defaults
    params = _expand_params(solver, params, timing)

    # try to get warnings from sampleset.info
    if warnings is None:
        warnings = sampleset.info.get('warnings')

    # construct problem stats
    problem_stats = _problem_stats(response=None, sampleset=sampleset,
                                   embedding_context=embedding_context)

    data = {
        "ready": True,
        "details": {
            "id": problem_id,
            "type": problem_type,
            "solver": solver.id,
            "label": sampleset.info.get('problem_label'),
        },
        "data": _problem_dict(solver_id, problem_type, problem_data, params, problem_stats),
        "answer": _answer_dict(solutions, active_variables, energies, num_occurrences, timing, num_variables),
        "unembedded_answer": _unembedded_answer_dict(sampleset),
        "warnings": _warnings(warnings),
        "rel": dict(solver=solver),
    }

    logger.trace("from_bqm_sampleset returned %r", data)

    return data


def from_objects(*args, **kwargs):
    """Based on positional argument types and keyword arguments, select the
    best adapter match and constructs the problem data.

    See :meth:`.from_qmi_response`, :meth:`.from_bqm_response`,
    :meth:`.from_bqm_sampleset` for details on possible arguments.
    """
    logger.debug("from_objects(*{!r}, **{!r})".format(args, kwargs))

    bqm_cls = BQM_CLASSES
    sampleset_cls = dimod.SampleSet
    sampler_cls = (dimod.Sampler, dimod.ComposedSampler)
    response_cls = dwave.cloud.Future
    is_embedding_ctx = lambda x: isinstance(x, abc.Mapping) and 'embedding' in x
    is_warnings = \
        lambda x: isinstance(x, abc.Sequence) and len(x) > 0 \
                  and isinstance(x[0], abc.Mapping) and 'message' in x[0]
    is_ising_problem = \
        lambda x: isinstance(x, abc.Sequence) and len(x) == 2 \
                and isinstance(x[0], (abc.Sequence, abc.Mapping)) \
                and isinstance(x[1], abc.Mapping)
    is_qubo_problem = \
        lambda x: isinstance(x, abc.Mapping) \
                  and all(isinstance(k, abc.Sequence) and len(k) == 2 for k in x)
    is_problem = lambda x: is_ising_problem(x) or is_qubo_problem(x)
    is_problem_id = lambda x: isinstance(x, str)

    bqms = list(filter(lambda arg: isinstance(arg, bqm_cls), args))
    samplesets = list(filter(lambda arg: isinstance(arg, sampleset_cls), args))
    samplers = list(filter(lambda arg: isinstance(arg, sampler_cls), args))
    responses = list(filter(lambda arg: isinstance(arg, response_cls), args))
    embedding_contexts = list(filter(is_embedding_ctx, args))
    warnings_candidates = list(filter(is_warnings, args))
    problems = list(filter(is_problem, args))
    problem_ids = list(filter(is_problem_id, args))

    maybe_pop = lambda ls: ls.pop() if len(ls) else None

    bqm = kwargs.get('bqm', maybe_pop(bqms))
    sampleset = kwargs.get('sampleset', maybe_pop(samplesets))
    sampler = kwargs.get('sampler', maybe_pop(samplers))
    response = kwargs.get('response', maybe_pop(responses))
    embedding_context = kwargs.get('embedding_context', maybe_pop(embedding_contexts))
    warnings = kwargs.get('warnings', maybe_pop(warnings_candidates))
    problem = kwargs.get('problem', maybe_pop(problems))
    problem_id = kwargs.get('problem_id', maybe_pop(problem_ids))

    # make sure the response is resolved
    if response is not None and not response.done():
        logger.debug("response not yet resolved; forcing resolve")
        response.result()

    # read problem_id from sampleset or response
    # TODO: resolve sampleset/response if they are not resolved yet?
    if problem_id is None and sampleset is not None:
        problem_id = sampleset.info.get('problem_id')
    if problem_id is None and response is not None:
        problem_id = response.id

    # read embedding_context and warnings from sampleset
    if sampleset is not None:
        embedding_context = sampleset.info.get('embedding_context', {})
        warnings = sampleset.info.get('warnings')

    logger.debug("from_objects detected {!r}".format(
        dict(bqm=bqm, sampleset=sampleset, sampler=sampler, response=response,
             embedding_context=embedding_context, warnings=warnings,
             problem=problem, problem_id=problem_id)))

    # ideally, low-level data capture has been enabled during sampling,
    # and we have the problem data:
    try:
        pd = storage.get_problem(problem_id)
    except KeyError:
        pd = None

    # in order of preference (most desirable form first):
    if pd is not None:
        # TODO: doublecheck qubos are visualized correctly
        problem = (pd.problem['linear'], pd.problem['quadratic'])
        return from_qmi_response(problem=problem, response=pd.response,
            embedding_context=embedding_context, warnings=warnings,
            params=pd.problem['params'], sampleset=sampleset)

    if problem is not None and response is not None:
        return from_qmi_response(
            problem=problem, response=response,
            embedding_context=embedding_context, warnings=warnings,
            sampleset=sampleset)

    if bqm is not None and embedding_context is not None and response is not None:
        return from_bqm_response(
            bqm=bqm, embedding_context=embedding_context,
            response=response, warnings=warnings, sampleset=sampleset)

    if bqm is not None and sampleset is not None and sampler is not None:
        return from_bqm_sampleset(
            bqm=bqm, sampleset=sampleset, sampler=sampler,
            embedding_context=embedding_context, warnings=warnings)

    raise ValueError(
        "invalid combination of arguments provided: if data capture not "
        "enabled, problem/response/solver have to be specified; "
        "also, make sure a structured problem is being inspected")
