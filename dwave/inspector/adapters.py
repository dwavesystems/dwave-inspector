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

import dimod
from dwave.cloud.utils import reformat_qubo_as_ising, uniform_get
from dwave.embedding import embed_bqm
from dwave.embedding.utils import edgelist_to_adjacency


class ProblemData(object):
    solver_id = None
    solver_data = None


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


def from_qmi_response(problem, response, embedding=None, warnings=None):
    """Construct problem data for visualization based on the low-level sampling
    problem definition and the low-level response.

    Args:
        problem ((list/dict, dict[(int, int), float]) or dict[(int, int), float]):
            Problem in Ising or QUBO form, conforming to solver graph.

        response (:class:`dwave.cloud.computation.Future`):
            Sampling response, as returned by the low-level sampling interface
            in the Cloud Client (e.g. :meth:`dwave.cloud.solver.sample_ising`
            for Ising problems).

        embedding (dict, optional):
            An embedding of logical problem onto the solver's graph.

        warnings (list[dict], optional):
            Optional list of warnings. Not implemented yet.

    """

    try:
        linear, quadratic = problem
    except:
        linear, quadratic = reformat_qubo_as_ising(qubo)

    solver = response.solver
    solver_id = solver.id
    solver_data = solver.data
    problem_type = response.problem_type

    variables = list(response.variables)
    num_variables = len(solver.variables)

    solutions = response['solutions']
    energies = response['energies']
    num_occurrences = response['num_occurrences']
    timing = response.timing

    # note: we can't use encode_problem_as_qp(solver, linear, quadratic) because
    # visualizer accepts decoded lists
    problem_data = {
        "format": "qp",         # SAPI non-conforming (nulls vs nans)
        "lin": [uniform_get(linear, v) for v in solver._encoding_qubits],
        "quad": [quadratic.get((q1,q2), 0)
                 for (q1,q2) in solver._encoding_couplers
                 if q1 in response.variables and q2 in response.variables]
    }

    # include optional embedding
    if embedding is not None:
        problem_data['embedding'] = embedding

    data = {
        "ready": True,
        "details": _details_dict(response),
        "data": _problem_dict(solver_id, problem_type, problem_data),
        "answer": _answer_dict(solutions, variables, energies, num_occurrences, timing, num_variables),

        # TODO
        "messages": [],
        "warnings": [],
    }

    return data


def from_bqm_response(bqm, embedding, response, warnings=None):
    """Construct problem data for visualization based on the unembedded BQM,
    the embedding used when submitting, and the low-level sampling response.

    Args:
        bqm (:class:`dimod.BinaryQuadraticModel`):
            Problem in logical (unembedded) space, given as a BQM.

        embedding (dict):
            An embedding of logical problem onto the solver's graph.

        response (:class:`dwave.cloud.computation.Future`):
            Sampling response, as returned by the low-level sampling interface
            in the Cloud Client (e.g. :meth:`dwave.cloud.solver.sample_ising`
            for Ising problems).

        warnings (list[dict], optional):
            Optional list of warnings. Not implemented yet.

    """

    solver = response.solver
    solver_id = solver.id
    problem_type = response.problem_type

    variables = list(response.variables)
    num_variables = len(solver.variables)

    def expand_sample(sample):
        m = dict(zip(variables, sample))
        return [int(m.get(v, 0)) for v in range(solver.num_qubits)]

    solutions = [expand_sample(sample) for sample in response.samples]
    energies = list(map(float, response.energies))
    num_occurrences = list(map(int, response.occurrences))
    timing = response.timing

    # bqm vartype must match response vartype
    if problem_type == "ising":
        bqm = bqm.change_vartype(dimod.SPIN, inplace=False)
    else:
        bqm = bqm.change_vartype(dimod.BINARY, inplace=False)

    # get embedded bqm
    source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]
    target_edgelist = solver.edges
    target_adjacency = edgelist_to_adjacency(target_edgelist)
    bqm_embedded = embed_bqm(bqm, embedding, target_adjacency)

    lin, quad = bqm_embedded.linear, bqm_embedded.quadratic
    problem_data = {
        "format": "qp",         # SAPI non-conforming (nulls vs nans)
        "lin": [lin.get(v) for v in solver._encoding_qubits],
        "quad": [quad.get((q1,q2), 0)
                 for (q1,q2) in solver._encoding_couplers
                 if q1 in variables and q2 in variables],
        "embedding": embedding
    }

    data = {
        "ready": True,
        "details": _details_dict(response),
        "data": _problem_dict(solver_id, problem_type, problem_data),
        "answer": _answer_dict(solutions, variables, energies, num_occurrences, timing, num_variables),

        # TODO
        "messages": [],
        "warnings": [],
    }

    return data


def from_bqm_sampleset(bqm, sampleset, sampler, embedding=None, warnings=None):
    """Construct problem data for visualization based on the BQM and sampleset
    in logical space (both unembedded).

    In order for the embedded problem/response to be reconstructed, an embedding
    is required in either the sampleset, or as a standalone argument.

    Args:
        bqm (:class:`dimod.BinaryQuadraticModel`):
            Problem in logical (unembedded) space, given as a BQM.

        sampleset (:class:`~dimod.sampleset.SampleSet`):
            Sampling response as a sampleset.

        sampler (:class:`~dimod.Sampler` or :class:`~dimod.ComposedSampler`):
            The :class:`~dwave.system.samplers.dwave_sampler.DWaveSampler`-
            derived sampler used to produce the sampleset off the bqm.

        embedding (dict, optional):
            An embedding of the logical problem onto the solver's graph. It is
            optional only if ``sampleset.info`` containes the embedding (see
            `return_embedding` argument of
            :meth:`~dwave.system.composites.embedding.EmbeddingComposite`).

        warnings (list[dict], optional):
            Optional list of warnings. Not implemented yet.

    """

    if not isinstance(sampler, dimod.Sampler):
        raise TypeError("dimod.Sampler instance expected for 'sampler'")

    if embedding is None:
        embedding = sampleset.info.get('embedding')
    if embedding is None:
        raise ValueError("embedding not given")

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
    bqm_embedded = embed_bqm(bqm, embedding, target_adjacency)

    variables = list(bqm_embedded.variables)
    num_variables = len(solver.variables)

    def expand_sample(sample):
        m = dict(zip(variables, sample))
        return [int(m.get(v, 0)) for v in range(solver.num_qubits)]

    solutions = [expand_sample(sample) for sample in sampleset.record.sample]
    energies = list(map(float, sampleset.record.energy))
    num_occurrences = list(map(int, sampleset.record.num_occurrences))
    timing = sampleset.info.get('timing')

    lin, quad = bqm_embedded.linear, bqm_embedded.quadratic
    problem_data = {
        "format": "qp",         # SAPI non-conforming (nulls vs nans)
        "lin": [lin.get(v) for v in solver._encoding_qubits],
        "quad": [quad.get((q1,q2), 0)
                 for (q1,q2) in solver._encoding_couplers
                 if q1 in variables and q2 in variables],
        "embedding": embedding
    }

    # problem id not available, auto-generate some
    problem_id = "local-%s" % uuid.uuid4()

    data = {
        "ready": True,
        "details": {
            "id": problem_id,
            "type": problem_type,
            "solver": solver.id
        },
        "data": _problem_dict(solver_id, problem_type, problem_data),
        "answer": _answer_dict(solutions, variables, energies, num_occurrences, timing, num_variables),

        # TODO
        "messages": [],
        "warnings": [],
    }

    return data
