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
import mock
import unittest
from decimal import Decimal
from fractions import Fraction
from functools import partial

import vcr
import numpy

import dimod
from dwave.cloud import Client
from dwave.cloud.solver import UnstructuredSolver
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.system.testing import MockDWaveSampler
from dwave.embedding import embed_bqm
from dwave.embedding.utils import edgelist_to_adjacency
from dwave.cloud.utils import reformat_qubo_as_ising, uniform_get, active_qubits

from dwave.inspector.adapters import (
    from_qmi_response, from_bqm_response, from_bqm_sampleset, from_objects,
    _validated_embedding)


rec = vcr.VCR(
    serializer='yaml',
    cassette_library_dir='tests/fixtures/cassettes',
    record_mode='none',
    match_on=['uri', 'method'],
    filter_headers=['x-auth-token'],
)

# minimal mock of an unstructured solver
unstructured_solver_mock = UnstructuredSolver(
    client=None,
    data={'id': 'mock',
          'properties': {'supported_problem_types': ['bqm']}})

# we can use a fake token because outbound requests are intercepted anyway
BrickedClient = partial(Client, token='fake')


@mock.patch('dwave.system.samplers.dwave_sampler.Client.from_config', BrickedClient)
class TestAdapters(unittest.TestCase):

    @rec.use_cassette('triangle-ising.yaml')
    def setUp(self):
        with BrickedClient() as client:
            self.solver = client.get_solver(qpu=True)

            self.ising = ({}, {'ab': 1, 'bc': 1, 'ca': 1})
            self.bqm = dimod.BQM.from_ising(*self.ising)
            self.embedding = {'a': [0], 'b': [4], 'c': [1, 5]}
            self.chain_strength = 1.0
            self.embedding_context = dict(embedding=self.embedding,
                                          chain_strength=self.chain_strength)

            target_edgelist = [[0, 4], [0, 5], [1, 4], [1, 5]]
            target_adjacency = edgelist_to_adjacency(target_edgelist)
            self.bqm_embedded = embed_bqm(self.bqm, self.embedding, target_adjacency,
                                          chain_strength=self.chain_strength)
            self.ising_embedded = self.bqm_embedded.to_ising()
            self.problem = self.ising_embedded[:2]

            self.params = dict(num_reads=100)

            # get the expected response (from VCR)
            self.response = self.solver.sample_ising(*self.problem, **self.params)

    def verify_data_encoding(self, problem, response, solver, params, data, embedding_context=None):
        # make sure data correct after JSON decoding (minus the 'rel' data)
        del data['rel']
        data = json.loads(json.dumps(data))

        # test structure
        self.assertIsInstance(data, dict)
        self.assertTrue(all(k in data for k in 'details data answer warnings'.split()))

        # .details
        self.assertIn('id', data['details'])
        self.assertEqual(data['details']['solver'], solver.id)

        # .problem
        self.assertEqual(data['data']['type'], response.problem_type)

        # .problem.params, smoke tests
        self.assertIn('params', data['data'])
        self.assertEqual(data['data']['params']['num_reads'], params['num_reads'])
        self.assertIn('annealing_time', data['data']['params'])
        self.assertIn('programming_thermalization', data['data']['params'])

        if response.problem_type == 'ising':
            linear, quadratic = problem
        elif response.problem_type == 'qubo':
            linear, quadratic = reformat_qubo_as_ising(problem)
        else:
            self.fail("Unknown problem type")

        active_variables = response['active_variables']
        problem_data = {
            "format": "qp",
            "lin": [uniform_get(linear, v, 0 if v in active_variables else None)
                    for v in solver._encoding_qubits],
            "quad": [quadratic.get((q1,q2), 0) + quadratic.get((q2,q1), 0)
                    for (q1,q2) in solver._encoding_couplers
                    if q1 in active_variables and q2 in active_variables]
        }
        if embedding_context is not None:
            problem_data['embedding'] = embedding_context['embedding']
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

    @rec.use_cassette('triangle-ising.yaml')
    def test_from_qmi_response__ising(self):
        """Inspector data is correctly encoded for a simple Ising triangle problem."""

        # sample
        with BrickedClient() as client:
            solver = client.get_solver(qpu=True)
            response = solver.sample_ising(*self.problem, **self.params)

        # convert
        data = from_qmi_response(self.problem, response, params=self.params)

        # validate data encoding
        self.verify_data_encoding(problem=self.problem, response=response,
                                  solver=solver, params=self.params, data=data)

    @rec.use_cassette('triangle-qubo.yaml')
    def test_from_qmi_response__qubo(self):
        """Inspector data is correctly encoded for a simple QUBO triangle problem."""

        # vars = (0, 1, 4, 5)
        # h = {}, J = {(0, 4): 1, (0, 5): 1, (1, 5): -1, (4, 1): 1}
        problem = {
            (0, 0): 0,   (0, 1): 0,    (0, 4): 0.5, (0, 5): 0.5,
            (1, 0): 0,   (1, 1): 0,    (1, 4): 0.5, (1, 5): -0.5,
            (4, 0): 0.5, (4, 1): 0.5,  (4, 4): 0,   (4, 5): 0,
            (5, 0): 0.5, (5, 1): -0.5, (5, 4): 0,   (5, 5): 0,
        }

        # sample
        with BrickedClient() as client:
            solver = client.get_solver(qpu=True)
            response = solver.sample_qubo(problem, **self.params)

        # convert
        data = from_qmi_response(problem, response, params=self.params)

        # validate data encoding
        self.verify_data_encoding(problem=problem, response=response,
                                  solver=solver, params=self.params, data=data)

    @rec.use_cassette('triangle-ising.yaml')
    def test_from_qmi_response__couplings_only(self):
        """Problem/solutions are correctly encoded when qubits are referenced via couplings only."""

        problem = ({}, self.ising_embedded[1])

        # sample
        with BrickedClient() as client:
            solver = client.get_solver(qpu=True)
            response = solver.sample_ising(*problem, **self.params)

        # convert
        data = from_qmi_response(problem, response, params=self.params)

        # validate data encoding
        self.verify_data_encoding(problem=problem, response=response,
                                  solver=solver, params=self.params, data=data)

    @rec.use_cassette('triangle-ising.yaml')
    def test_from_qmi_response__problem_encoding(self):
        """Problem data is serialized even when it uses non-standard types (like numpy.int64)."""

        # `self.problem` == (
        #   {0: 0.0, 4: 0.0, 1: 0.0, 5: 0.0},
        #   {(0, 4): 1.0, (0, 5): 1.0, (4, 1): 1.0, (1, 5): -1.0}
        # )
        h = {
            0: numpy.int64(0),
            4: numpy.double(0),
            1: numpy.int8(0),
            5: Decimal('0'),
        }
        J = {
            (0, 4): numpy.float128(1),
            (0, 5): Decimal('1'),
            (4, 1): Fraction(2, 2),
            (1, 5): numpy.int32(-1),
        }
        problem = (h, J)

        # sample
        with BrickedClient() as client:
            solver = client.get_solver(qpu=True)
            response = solver.sample_ising(*problem, **self.params)

        # convert
        data = from_qmi_response(problem, response, params=self.params)

        # validate data encoding
        self.verify_data_encoding(problem=problem, response=response,
                                  solver=solver, params=self.params, data=data)

    @rec.use_cassette('triangle-ising.yaml')
    def _test_from_bqm_response(self, bqm):
        # sample
        with BrickedClient() as client:
            solver = client.get_solver(qpu=True)
            response = solver.sample_ising(*self.problem, **self.params)

            # induce sampleset production in response, to test serialization of
            # sampleset-provided data, like `num_occurrences` (an numpy.ndarray)
            # NOTE: `dwave.cloud.computation.Future.occurrences` et al. will
            # favorize returning data from a sampleset, if it's present, instead
            # of returning raw SAPI data
            _ = response.sampleset

        # convert
        data = from_bqm_response(bqm, self.embedding_context, response,
                                 params=self.params)

        # validate data encoding
        self.verify_data_encoding(problem=self.problem, response=response,
                                  solver=solver, params=self.params, data=data,
                                  embedding_context=self.embedding_context)

    def test_from_old_bqm_response(self):
        self._test_from_bqm_response(self.bqm)

    @unittest.skipUnless('AdjVectorBQM' in dir(dimod), 'requires dimod.AdjVectorBQM')
    def test_from_AdjVectorBQM_response(self):
        # cast dict bqm to AdjVectorBQM
        bqm = dimod.as_bqm(self.bqm, cls=[dimod.AdjVectorBQM])

        self._test_from_bqm_response(bqm)

    # NOTE: omit AdjArrayBQM until https://github.com/dwavesystems/dwave-system/issues/261 is fixed
    # @unittest.skipUnless('AdjArrayBQM' in dir(dimod), 'requires dimod.AdjArrayBQM')
    # def test_from_AdjArrayBQM_response(self):
    #     # cast dict bqm to AdjArrayBQM
    #     bqm = dimod.as_bqm(self.bqm, cls=[dimod.AdjArrayBQM])
    #
    #     self._test_from_bqm_response(bqm)

    @unittest.skipUnless('AdjDictBQM' in dir(dimod), 'requires dimod.AdjDictBQM')
    def test_from_AdjDictBQM_response(self):
        # cast dict bqm to AdjDictBQM
        bqm = dimod.as_bqm(self.bqm, cls=[dimod.AdjDictBQM])

        self._test_from_bqm_response(bqm)

    @unittest.skipUnless('AdjMapBQM' in dir(dimod), 'requires dimod.AdjMapBQM')
    def test_from_AdjMapBQM_response(self):
        # cast dict bqm to AdjMapBQM
        bqm = dimod.as_bqm(self.bqm, cls=[dimod.AdjMapBQM])

        self._test_from_bqm_response(bqm)

    @rec.use_cassette('triangle-ising.yaml')
    def _test_from_bqm_sampleset(self, bqm):
        # sample
        qpu = DWaveSampler(solver=dict(qpu=True))
        sampler = FixedEmbeddingComposite(qpu, self.embedding)
        sampleset = sampler.sample(bqm, return_embedding=True, **self.params)

        # convert
        data = from_bqm_sampleset(bqm, sampleset, sampler, params=self.params)

        # construct (unembedded) response with chain breaks resolved
        # NOTE: for bqm/sampleset adapter, this is the best we can expect :(

        # inverse the embedding
        var_to_idx = {var: idx for idx, var in enumerate(sampleset.variables)}
        unembedding = {q: var_to_idx[v] for v, qs in self.embedding.items() for q in qs}

        # embed sampleset
        solutions_without_chain_breaks = [
            [int(sample[unembedding[q]]) if q in unembedding else val
                for q, val in enumerate(solution)]
            for solution, sample in zip(
                self.response['solutions'], sampleset.record.sample)]

        with mock.patch.dict(self.response._result,
                             {'solutions': solutions_without_chain_breaks}):

            # validate data encoding
            self.verify_data_encoding(problem=self.problem, response=self.response,
                                      solver=self.solver, params=self.params, data=data,
                                      embedding_context=self.embedding_context)

    def test_from_old_bqm_sampleset(self):
        self._test_from_bqm_sampleset(self.bqm)

    @unittest.skipUnless('AdjVectorBQM' in dir(dimod), 'requires dimod.AdjVectorBQM')
    def test_from_AdjVectorBQM_sampleset(self):
        # cast dict bqm to AdjVectorBQM
        bqm = dimod.as_bqm(self.bqm, cls=[dimod.AdjVectorBQM])

        self._test_from_bqm_sampleset(bqm)

    # NOTE: omit AdjArrayBQM until https://github.com/dwavesystems/dwave-system/issues/261 is fixed
    # @unittest.skipUnless('AdjArrayBQM' in dir(dimod), 'requires dimod.AdjArrayBQM')
    # def test_from_AdjArrayBQM_sampleset(self):
    #     # cast dict bqm to AdjArrayBQM
    #     bqm = dimod.as_bqm(self.bqm, cls=[dimod.AdjArrayBQM])
    #
    #     self._test_from_bqm_sampleset(bqm)

    @unittest.skipUnless('AdjDictBQM' in dir(dimod), 'requires dimod.AdjDictBQM')
    def test_from_AdjDictBQM_sampleset(self):
        # cast dict bqm to AdjDictBQM
        bqm = dimod.as_bqm(self.bqm, cls=[dimod.AdjDictBQM])

        self._test_from_bqm_sampleset(bqm)

    @unittest.skipUnless('AdjMapBQM' in dir(dimod), 'requires dimod.AdjMapBQM')
    def test_from_AdjMapBQM_sampleset(self):
        # cast dict bqm to AdjMapBQM
        bqm = dimod.as_bqm(self.bqm, cls=[dimod.AdjMapBQM])

        self._test_from_bqm_sampleset(bqm)

    @mock.patch('dwave.inspector.adapters.from_qmi_response', return_value='qmi_response')
    @mock.patch('dwave.inspector.adapters.from_bqm_response', return_value='bqm_response')
    @mock.patch('dwave.inspector.adapters.from_bqm_sampleset', return_value='bqm_sampleset')
    def test_from_objects(self, m1, m2, m3):
        # qmi
        self.assertEqual(from_objects(self.problem, self.response), 'qmi_response')
        self.assertEqual(from_objects(self.response, self.problem), 'qmi_response')
        self.assertEqual(from_objects(response=self.response, problem=self.problem), 'qmi_response')
        self.assertEqual(from_objects(self.embedding_context, response=self.response, problem=self.problem), 'qmi_response')
        self.assertEqual(from_objects(self.bqm, response=self.response, problem=self.problem), 'qmi_response')
        self.assertEqual(from_objects({(0, 0): 1, (0, 1): 0}, self.response), 'qmi_response')

        # reconstruction directly from problem_id
        self.assertEqual(from_objects(self.response.id), 'qmi_response')

        # qmi takes precedence
        self.assertEqual(from_objects(self.bqm, self.embedding_context, response=self.response, problem=self.problem), 'qmi_response')

        # bqm/response -> with problem_id in response ==> qmi takes precedence
        self.assertEqual(from_objects(self.response, self.bqm, self.embedding_context), 'qmi_response')
        self.assertEqual(from_objects(self.embedding_context, response=self.response, bqm=self.bqm), 'qmi_response')
        self.assertEqual(from_objects(response=self.response, bqm=self.bqm, embedding_context=self.embedding_context), 'qmi_response')

        # bqm/response -> without problem_id in response
        self.response.id = None
        self.assertEqual(from_objects(self.response, self.bqm, self.embedding_context), 'bqm_response')
        self.assertEqual(from_objects(self.embedding_context, response=self.response, bqm=self.bqm), 'bqm_response')
        self.assertEqual(from_objects(response=self.response, bqm=self.bqm, embedding_context=self.embedding_context), 'bqm_response')

        # bqm/sampleset
        sampler = MockDWaveSampler()
        sampleset = self.response.sampleset
        warnings = [{'message': 'test'}]
        self.assertEqual(from_objects(self.bqm, sampleset, sampler), 'bqm_sampleset')
        self.assertEqual(from_objects(self.bqm, sampleset, sampler, warnings), 'bqm_sampleset')
        self.assertEqual(from_objects(sampler, warnings, sampleset=sampleset, bqm=self.bqm), 'bqm_sampleset')

    @rec.use_cassette('triangle-ising.yaml')
    def test_solver_type_validation(self):
        """All data adapters should fail on non-StructuredSolvers."""

        # sample
        with BrickedClient() as client:
            solver = client.get_solver(qpu=True)
            response = solver.sample_ising(*self.problem, **self.params)

        # resolve it before we mangle with it
        response.result()
        # change solver to unstructured to test solver validation
        response.solver = unstructured_solver_mock

        # ensure `from_qmi_response` adapter fails on unstructured solver
        with self.assertRaises(TypeError):
            from_qmi_response(self.problem, response, params=self.params)

        # ensure `from_bqm_response` adapter fails on unstructured solver
        with self.assertRaises(TypeError):
            from_bqm_response(
                self.bqm, self.embedding_context, response, params=self.params)

    @rec.use_cassette('triangle-ising.yaml')
    def test_sampler_type_validation(self):
        """All data adapters should fail on non-StructuredSolvers."""

        # sample
        qpu = DWaveSampler(solver=dict(qpu=True))
        sampler = FixedEmbeddingComposite(qpu, self.embedding)
        sampleset = sampler.sample(self.bqm, return_embedding=True, **self.params)

        # resolve it before we mangle with it
        sampleset.info['problem_id']
        # change solver to unstructured to test solver validation
        sampler.child.solver = unstructured_solver_mock

        # ensure `from_bqm_sampleset` adapter fails on unstructured solver
        with self.assertRaises(TypeError):
            from_bqm_sampleset(self.bqm, sampleset, sampler, params=self.params)

    @rec.use_cassette('triangle-ising.yaml')
    def test_solver_graph_validation(self):
        """All data adapters should fail on non-Chimera/Pegasus solvers."""

        # sample
        with BrickedClient() as client:
            solver = client.get_solver(qpu=True)
            response = solver.sample_ising(*self.problem, **self.params)

        # resolve it before we mangle with it
        response.result()
        # change solver topology to non-chimera/pegasus to test solver validation
        response.solver.properties['topology']['type'] = 'unknown'

        # ensure `from_qmi_response` adapter fails on unstructured solver
        with self.assertRaises(TypeError):
            from_qmi_response(self.problem, response, params=self.params)

        # ensure `from_bqm_response` adapter fails on unstructured solver
        with self.assertRaises(TypeError):
            from_bqm_response(
                self.bqm, self.embedding_context, response, params=self.params)

    @rec.use_cassette('triangle-ising.yaml')
    def test_sampler_graph_validation(self):
        """All data adapters should fail on non-Chimera/Pegasus solvers."""

        # sample
        qpu = DWaveSampler(solver=dict(qpu=True))
        sampler = FixedEmbeddingComposite(qpu, self.embedding)
        sampleset = sampler.sample(self.bqm, return_embedding=True, **self.params)

        # resolve it before we mangle with it
        sampleset.info['problem_id']
        # change solver topology to non-chimera/pegasus to test solver validation
        sampler.child.solver.properties['topology']['type'] = 'unknown'

        # ensure `from_bqm_sampleset` adapter fails on unstructured solver
        with self.assertRaises(TypeError):
            from_bqm_sampleset(self.bqm, sampleset, sampler, params=self.params)

    def test_embedding_validation(self):
        # chains can be non-lists

        # like sets (issue 19)
        emb = {'a': {0}, 'b': {2,1}}
        validated = _validated_embedding(emb)
        self.assertDictEqual(validated, {'a': [0], 'b': [1,2]})

        # or numpy arrays
        emb = {'a': numpy.array([1,2])}
        validated = _validated_embedding(emb)
        self.assertDictEqual(validated, {'a': [1,2]})

        # or other iterables
        emb = {'a': {0: 1, 1: 2}}
        validated = _validated_embedding(emb)
        self.assertDictEqual(validated, {'a': [0,1]})

        # source variables can be non-strings
        emb = {0: [0], 1: [1]}
        validated = _validated_embedding(emb)
        self.assertDictEqual(validated, {"0": [0], "1": [1]})

        # target variables can be non-integers
        emb = {'a': [numpy.int64(0)]}
        validated = _validated_embedding(emb)
        self.assertDictEqual(validated, {'a': [0]})

        # invalid embedding data structure
        with self.assertRaises(ValueError):
            _validated_embedding([['a'], [1,2]])

        with self.assertRaises(ValueError):
            _validated_embedding("a")

        # validate overlapping chains fail (issue #67)
        with self.assertRaises(ValueError):
            _validated_embedding({'a': [0, 4], 'b': [4, 5]})

        with self.assertRaises(ValueError):
            _validated_embedding({'a': [0, 1, 2], 'b': [1]})

        with self.assertRaises(ValueError):
            _validated_embedding({'a': [0, 1, 2], 'b': [3], 'c': [3, 0]})

        with self.assertRaises(ValueError):
            _validated_embedding({0: [0, 4], 1: [4, 3], 2: [3, 7], 3: [7, 0]})
