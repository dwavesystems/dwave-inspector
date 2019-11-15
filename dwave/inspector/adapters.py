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

import dimod
from dwave.embedding import embed_bqm
from dwave.embedding.utils import edgelist_to_adjacency


def from_objects(bqm=None, embedding=None, response=None, warnings=None):
    if response is None:
        raise ValueError("response not yet optional")

    if embedding is None:
        raise ValueError("embedding not yet optional")

    solver = response.solver
    solver_id = solver.id
    problem_type = response.problem_type

    variables = list(response.variables)
    num_qubits = len(solver._encoding_qubits)

    def expand_sample(sample):
        m = dict(zip(variables, sample))
        return [int(m.get(v, 0)) for v in range(solver.num_qubits)]

    solutions = [expand_sample(sample) for sample in response.samples]
    energies = list(map(float, response.energies))
    num_occurrences = list(map(int, response.occurrences))

    # bqm vartype must match response vartype
    if problem_type == "ising":
        bqm = bqm.change_vartype(dimod.SPIN, inplace=False)
    else:
        bqm = bqm.change_vartype(dimod.BINARY, inplace=False)

    # get embedded bqm
    # XXX: embedding required
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
                 if q1 in variables and q2 in variables]
    }

    if embedding is not None:
        problem_data['embedding'] = embedding

    data = {
        "ready": True,
        "details": {
            "status": response.remote_status,
            "id": response.id,
            "solver": solver_id,
            "type": problem_type,
            "submitted_on": response.time_received.isoformat(),
            "solved_on": response.time_solved.isoformat(),
        },

        # TODO
        "messages": [],

        "answer": {
            "format": "qp",
            "solutions": solutions,             # cloud-client non-conforming
            "active_variables": variables,
            "energies": energies,
            "num_occurrences": num_occurrences,
            "timing": response.timing,
            "num_variables": num_qubits         # len(variables)     # SAPI non-conforming
        },

        # TODO
        "warnings": [],

        # problem definition
        "data": {
            "solver": solver_id,
            "type": problem_type,
            # XXX: only available in the submitted request
            "params": {},
            "data": problem_data
        }
    }

    return data
