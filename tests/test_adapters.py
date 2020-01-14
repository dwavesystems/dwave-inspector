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

import json
import unittest

import vcr

import dimod
from dwave.cloud import Client
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.embedding import embed_bqm
from dwave.embedding.utils import edgelist_to_adjacency
from dwave.cloud.utils import active_qubits, uniform_get

from dwave.inspector.adapters import (
    from_qmi_response, from_bqm_response, from_bqm_sampleset)


rec = vcr.VCR(
    serializer='yaml',
    cassette_library_dir='tests/fixtures/cassettes',
    record_mode='none',
    match_on=['uri', 'method'],
    filter_headers=['x-auth-token'],
)


class TestAdapters(unittest.TestCase):

    @rec.use_cassette('triangle.yaml')
    def setUp(self):
        with Client.from_config() as client:
            self.solver = client.get_solver(qpu=True)

        self.ising = ({}, {'ab': 1, 'bc': 1, 'ca': 1})
        self.bqm = dimod.BQM.from_ising(*self.ising)
        self.embedding = {'a': [0], 'b': [4], 'c': [1, 5]}

        target_edgelist = [[0, 4], [0, 5], [1, 4], [1, 5]]
        target_adjacency = edgelist_to_adjacency(target_edgelist)
        self.bqm_embedded = embed_bqm(self.bqm, self.embedding, target_adjacency)
        self.ising_embedded = self.bqm_embedded.to_ising()
        self.problem = self.ising_embedded[:2]

        self.params = dict(num_reads=100)

    def verify_data_encoding(self, problem, response, solver, params, data):
        # make sure data correct after JSON decoding
        data = json.loads(json.dumps(data))

        # test structure
        self.assertIsInstance(data, dict)
        self.assertTrue(all(k in data for k in 'details data answer warnings'.split()))

        # .details
        self.assertEqual(data['details']['id'], response.id)
        self.assertEqual(data['details']['solver'], solver.id)

        # .problem
        self.assertEqual(data['data']['type'], response.problem_type)

        # XXX: params not supported yet
        #self.assertEqual(data['data']['params'], params)

        linear, quadratic = problem
        active_variables = response['active_variables']
        problem_data = {
            "format": "qp",
            "lin": [uniform_get(linear, v, 0 if v in active_variables else None)
                    for v in solver._encoding_qubits],
            "quad": [quadratic.get((q1,q2), 0) + quadratic.get((q2,q1), 0)
                    for (q1,q2) in solver._encoding_couplers
                    if q1 in active_variables and q2 in active_variables]
        }
        self.assertEqual(data['data']['data'], problem_data)

        # .answer
        self.assertEqual(sum(data['answer']['num_occurrences']), params['num_reads'])
        self.assertEqual(data['answer']['num_occurrences'], response['num_occurrences'])
        self.assertEqual(data['answer']['num_variables'], response['num_variables'])
        self.assertEqual(data['answer']['active_variables'], active_variables)
        solutions = [[sol[idx] for idx in active_variables ] for sol in response['solutions']]
        self.assertEqual(data['answer']['solutions'], solutions)
        self.assertEqual(data['answer']['energies'], response['energies'])
        self.assertEqual(data['answer']['timing'], response['timing'])

    @rec.use_cassette('triangle.yaml')
    def test_from_qmi_response(self):
        """Inspector data is correctly encoded for a simple Ising triangle problem."""

        # sample
        with Client.from_config() as client:
            solver = client.get_solver(qpu=True)
            response = solver.sample_ising(*self.problem, **self.params)

        # convert
        data = from_qmi_response(self.problem, response)

        # validate data encoding
        self.verify_data_encoding(problem=self.problem, response=response,
                                  solver=solver, params=self.params, data=data)

    @rec.use_cassette('triangle.yaml')
    def test_from_qmi_response__couplings_only(self):
        """Problem/solutions are correctly encoded when qubits are referenced via couplings only."""

        problem = ({}, self.ising_embedded[1])

        # sample
        with Client.from_config() as client:
            solver = client.get_solver(qpu=True)
            response = solver.sample_ising(*problem, **self.params)

        # convert
        data = from_qmi_response(problem, response)

        # validate data encoding
        self.verify_data_encoding(problem=self.problem, response=response,
                                  solver=solver, params=self.params, data=data)

    @rec.use_cassette('triangle.yaml')
    def test_from_bqm_response(self):
        # sample
        with Client.from_config() as client:
            solver = client.get_solver(qpu=True)
            response = solver.sample_ising(*self.problem, num_reads=100)

        # convert
        data = from_bqm_response(self.bqm, self.embedding, response)

        # validate
        self.assertEqual(data['details']['solver'], solver.id)
        self.assertEqual(sum(data['answer']['num_occurrences']), 100)

    @rec.use_cassette('triangle.yaml')
    def test_from_bqm_sampleset(self):
        # sample
        qpu = DWaveSampler(solver=dict(qpu=True))
        sampler = FixedEmbeddingComposite(qpu, self.embedding)
        sampleset = sampler.sample(self.bqm, num_reads=100, return_embedding=True)

        # convert
        data = from_bqm_sampleset(self.bqm, sampleset, sampler, self.embedding)

        # validate
        self.assertEqual(data['details']['solver'], qpu.solver.id)
        self.assertEqual(sum(data['answer']['num_occurrences']), 100)

    def test_from_objects(self):
        pass
