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

from __future__ import absolute_import

import uuid
import logging
from operator import itemgetter
from collections import abc

import dimod
import dimod.views.bqm
import dwave.cloud
from dwave.cloud.utils import reformat_qubo_as_ising, uniform_get, active_qubits
from dwave.cloud.events import add_handler
from dwave.embedding import embed_bqm
from dwave.embedding.utils import edgelist_to_adjacency
from dwave.system.composites import EmbeddingComposite
from dwave.system.warnings import WarningAction

from dwave.inspector import storage

__all__ = [
    'from_qmi_response',
    'from_bqm_response',
    'from_bqm_sampleset',
    'from_objects',
    'enable_data_capture',
]

logger = logging.getLogger(__name__)


def enable_data_capture():
    """Enable logging of submitted problems/answers, embedding parameters,
    embedding/sampling warnings, etc. across the Ocean stack.
    """

    def capture_qmi_response(event, obj, args, return_value):
        logger.debug("{!s}(obj={!r}, args={!r}, return_value={!r})".format(
            event, obj, args, return_value))
        try:
            storage.add_problem(problem=args, solver=obj, response=return_value)
        except Exception as e:
            logger.error('Failed to store problem with: %r', e)

    # subscribe to problems sampled and results returned in the cloud client
    add_handler('after_sample', capture_qmi_response)

    # save all warnings during embedding by default (equivalent to setting
    # `warnings=WarningAction.SAVE` during sampling)
    EmbeddingComposite.warnings_default = WarningAction.SAVE

    # return embedding map/parameters (equivalent to setting
    # `return_embedding=True` during sampling)
    EmbeddingComposite.return_embedding_default = True

    logger.debug("Data capture enabled for embeddings, warnings and QMIs")


def _answer_dict(solutions, active_variables, energies, num_occurrences, timing, num_variables):
    return {
        "format": "qp",
        "solutions": solutions,
        "active_variables": active_variables,
        "energies": energies,
        "num_occurrences": num_occurrences,
        "timing": timing,
        "num_variables": num_variables
    }

def _problem_dict(solver_id, problem_type, problem_data, params=None):
    return {
        "solver": solver_id,
        "type": problem_type,
        "params": params if params is not None else {},
        "data": problem_data
    }

def _details_dict(response):
    return {
        "id": response.id,
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
    data = warnings.copy()
    for warning in data:
        if issubclass(warning['type'], Warning):
            warning.update(type=warning['type'].__name__)
    return data

def _validated_embedding(emb):
    "Basic types validation/conversion."

    try:
        keys = map(str, emb.keys())
        values = map(list, emb.values())
        return dict(zip(keys, values))

    except Exception as e:
        msg = "invalid embedding structure"
        logger.warning(msg)
        raise ValueError(msg)


def from_qmi_response(problem, response, embedding_context=None, warnings=None,
                      params=None):
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
            in the Cloud Client (e.g. :meth:`dwave.cloud.solver.sample_ising`
            for Ising problems).

        embedding_context (dict, optional):
            A map containing an embedding of logical problem onto the
            solver's graph (the ``embedding`` key) and embedding parameters
            used (e.g. ``chain_strength``).

        warnings (list[dict], optional):
            Optional list of warnings.

        params (dict, optional):
            Sampling parameters used.

    """
    logger.debug("from_qmi_response({!r})".format(
        dict(problem=problem, response=response, response_energies=response['energies'],
             embedding_context=embedding_context, warnings=warnings, params=params)))

    try:
        linear, quadratic = problem
    except:
        linear, quadratic = reformat_qubo_as_ising(problem)

    # make sure lin/quad are not dimod views (that handle directed edges)
    if isinstance(linear, dimod.views.bqm.BQMView):
        linear = dict(linear)
    if isinstance(quadratic, dimod.views.bqm.BQMView):
        quadratic = dict(quadratic)

    solver = response.solver
    solver_id = solver.id
    solver_data = solver.data
    problem_type = response.problem_type

    variables = list(response.variables)
    active = active_qubits(linear, quadratic)

    # sanity check
    active_variables = response['active_variables']
    assert set(active) == set(active_variables)

    solutions = list(map(itemgetter(*active_variables), response['solutions']))
    energies = response['energies']
    num_occurrences = response['num_occurrences']
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
        problem_data['embedding'] = \
            _validated_embedding(embedding_context['embedding'])

    # try to reconstruct sampling params
    if params is None:
        params = {'num_reads': sum(num_occurrences)}

    data = {
        "ready": True,
        "details": _details_dict(response),
        "data": _problem_dict(solver_id, problem_type, problem_data, params),
        "answer": _answer_dict(solutions, active_variables, energies, num_occurrences, timing, num_variables),
        "warnings": _warnings(warnings),

        # TODO
        "messages": [],
    }

    logger.trace("from_qmi_response returned %r", data)

    return data


def from_bqm_response(bqm, embedding_context, response, warnings=None,
                      params=None):
    """Construct problem data for visualization based on the unembedded BQM,
    the embedding used when submitting, and the low-level sampling response.

    Args:
        bqm (:class:`dimod.BinaryQuadraticModel`):
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

    """
    logger.debug("from_bqm_response({!r})".format(
        dict(bqm=bqm, response=response, response_energies=response['energies'],
             embedding_context=embedding_context, warnings=warnings, params=params)))

    solver = response.solver
    solver_id = solver.id
    problem_type = response.problem_type

    active_variables = response['active_variables']
    active = set(active_variables)

    solutions = list(map(itemgetter(*active_variables), response['solutions']))
    energies = response['energies']
    num_occurrences = response['num_occurrences']
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
    chain_strength = embedding_context.get('chain_strength', 1.0)
    chain_break_method = embedding_context.get('chain_break_method')

    # get embedded bqm
    source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]
    target_edgelist = solver.edges
    target_adjacency = edgelist_to_adjacency(target_edgelist)
    bqm_embedded = embed_bqm(bqm, embedding, target_adjacency,
                             chain_strength=chain_strength,
                             smear_vartype=dimod.SPIN)

    linear, quadratic, offset = bqm_embedded.to_ising()
    problem_data = {
        "format": "qp",         # SAPI non-conforming (nulls vs nans)
        "lin": [uniform_get(linear, v, 0 if v in active else None)
                for v in solver._encoding_qubits],
        "quad": [quadratic.get((q1,q2), 0) + quadratic.get((q2,q1), 0)
                 for (q1,q2) in solver._encoding_couplers
                 if q1 in active and q2 in active],
        "embedding": _validated_embedding(embedding)
    }

    # try to reconstruct sampling params
    if params is None:
        params = {'num_reads': sum(num_occurrences)}

    # TODO: if warnings are missing, calculate them here (since we have the
    # low-level response)

    data = {
        "ready": True,
        "details": _details_dict(response),
        "data": _problem_dict(solver_id, problem_type, problem_data, params),
        "answer": _answer_dict(solutions, active_variables, energies, num_occurrences, timing, num_variables),
        "warnings": _warnings(warnings),

        # TODO
        "messages": [],
    }

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
        bqm (:class:`dimod.BinaryQuadraticModel`):
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
    chain_strength = embedding_context.get('chain_strength', 1.0)
    chain_break_method = embedding_context.get('chain_break_method')

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
    solver_id = solver.id
    problem_type = "ising" if sampleset.vartype is dimod.SPIN else "qubo"

    # bqm vartype must match sampleset vartype
    if bqm.vartype is not sampleset.vartype:
        bqm = bqm.change_vartype(sampleset.vartype, inplace=False)

    # get embedded bqm
    source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]
    target_edgelist = solver.edges
    target_adjacency = edgelist_to_adjacency(target_edgelist)
    bqm_embedded = embed_bqm(bqm, embedding, target_adjacency,
                             chain_strength=chain_strength,
                             smear_vartype=dimod.SPIN)

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
        "embedding": _validated_embedding(embedding)
    }

    # try to get problem id. if not available, auto-generate one
    problem_id = sampleset.info.get('problem_id')
    if problem_id is None:
        problem_id = "local-%s" % uuid.uuid4()

    # try to reconstruct sampling params
    if params is None:
        params = {'num_reads': sum(num_occurrences)}

    # try to get warnings from sampleset.info
    if warnings is None:
        warnings = sampleset.info.get('warnings')

    data = {
        "ready": True,
        "details": {
            "id": problem_id,
            "type": problem_type,
            "solver": solver.id
        },
        "data": _problem_dict(solver_id, problem_type, problem_data, params),
        "answer": _answer_dict(solutions, active_variables, energies, num_occurrences, timing, num_variables),
        "warnings": _warnings(warnings),

        # TODO:
        "messages": [],
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

    bqm_cls = dimod.BinaryQuadraticModel
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

    # read problem_id from sampleset or response
    if sampleset is not None:
        problem_id = sampleset.info.get('problem_id')
    if response is not None:
        problem_id = response.id

    # read embedding_context and warnings from sampleset
    if sampleset is not None:
        embedding_context = sampleset.info.get('embedding_context', {})
        warnings = sampleset.info.get('warnings')

    logger.debug("from_objects detected {!r}".format(
        dict(bqm=bqm, sampleset=sampleset, sampler=sampler, response=response,
             embedding_context=embedding_context, warnings=warnings,
             problem=problem, problem_id=problem_id)))

    # in order of preference (most desirable form first):
    if problem_id is not None:
        pd = storage.get_problem(problem_id)
        # TODO: fix ising/qubo forms
        # XXX: for now assume type=='ising'!
        problem = (pd.problem['linear'], pd.problem['quadratic'])
        return from_qmi_response(problem=problem, response=pd.response,
            embedding_context=embedding_context, warnings=warnings,
            params=pd.problem['params'])

    if problem is not None and response is not None:
        return from_qmi_response(
            problem=problem, response=response,
            embedding_context=embedding_context, warnings=warnings)

    if bqm is not None and embedding_context is not None and response is not None:
        return from_bqm_response(
            bqm=bqm, embedding_context=embedding_context,
            response=response, warnings=warnings)

    if bqm is not None and sampleset is not None and sampler is not None:
        return from_bqm_sampleset(
            bqm=bqm, sampleset=sampleset, sampler=sampler,
            embedding_context=embedding_context, warnings=warnings)

    raise ValueError("invalid combination of arguments")
