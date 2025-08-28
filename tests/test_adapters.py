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
from unittest import mock
from decimal import Decimal
from fractions import Fraction

import numpy

import dimod
from dwave.cloud.solver import BQMSolver
from dwave.cloud.utils.qubo import reformat_qubo_as_ising
from dwave.cloud.testing.mocks import hybrid_bqm_solver_data
from dwave.embedding import embed_bqm
from dwave.embedding.utils import edgelist_to_adjacency
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.system.testing import MockDWaveSampler

from dwave.inspector.adapters import (
    from_qmi_response, from_bqm_response, from_bqm_sampleset, from_objects,
    _validated_embedding, _get_solver_id)

from tests import BrickedClient, sapi_vcr as rec


@mock.patch('dwave.system.samplers.dwave_sampler.Client.from_config', BrickedClient)
class TestAdapters(unittest.TestCase):

    @rec.use_cassette('triangle-ising.yaml')
    @classmethod
    def setUpClass(cls):
        with BrickedClient() as client:
            cls.solver = client.get_solver()

            # we used a zephyr qpu (Advantage2_system1.5) when recording the fixture
            cls.ising = ({}, {'ab': 1, 'bc': 1, 'ca': 1})
            cls.bqm = dimod.BQM.from_ising(*cls.ising)
            cls.embedding = {'a': [0], 'b': [1], 'c': [12]}
            cls.chain_strength = 1.0
            cls.embedding_context = dict(embedding=cls.embedding,
                                         chain_strength=cls.chain_strength)

            target_edgelist = [[0, 1], [0, 12], [1, 12]]
            target_adjacency = edgelist_to_adjacency(target_edgelist)
            cls.bqm_embedded = embed_bqm(cls.bqm, cls.embedding, target_adjacency,
                                         chain_strength=cls.chain_strength)
            cls.ising_embedded = cls.bqm_embedded.to_ising()
            cls.problem = cls.ising_embedded[:2]

            cls.params = dict(num_reads=100)
            cls.label = "pretty-label"

            # get the expected response (from VCR); make sure it's resolved
            cls.response = cls.solver.sample_ising(*cls.problem, **cls.params)
            cls.response.wait()

    def verify_data_encoding(self, problem, response, solver, params, data, embedding_context=None):
        # avoid persistent data modification
        data = data.copy()

        # make sure data correct after JSON decoding (minus the 'rel' data)
        del data['rel']
        data = json.loads(json.dumps(data))

        # test structure
        self.assertIsInstance(data, dict)
        self.assertTrue(all(k in data for k in 'details data answer warnings'.split()))

        # .details
        self.assertIn('id', data['details'])
        self.assertIn('label', data['details'])
        self.assertEqual(data['details']['solver'], _get_solver_id(solver))

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

        # make sure BQM views are converted to plain dicts, to prevent bias doubling
        if not isinstance(linear, dict):
            linear = dict(linear)
        if not isinstance(quadratic, dict):
            quadratic = dict(quadratic)

        active_variables = response['active_variables']
        problem_data = {
            "format": "qp",
            "lin": [linear.get(v, 0 if v in active_variables else None)
                    for v in solver._encoding_qubits],
            "quad": [quadratic.get((q1,q2), 0) + quadratic.get((q2,q1), 0)
                    for (q1,q2) in solver._encoding_couplers
                    if q1 in active_variables and q2 in active_variables]
        }
        if embedding_context is not None:
            problem_data['embedding'] = embedding_context['embedding']
        self.assertDictEqual(data['data']['data'], problem_data)

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
            solver = client.get_solver()
            response = solver.sample_ising(*self.problem, **self.params)
            response.wait()

        # convert
        data = from_qmi_response(self.problem, response, params=self.params)

        # validate data encoding
        self.verify_data_encoding(problem=self.problem, response=response,
                                  solver=solver, params=self.params, data=data)

    @rec.use_cassette('triangle-ising.yaml')
    def test_from_qmi_response__ising_bqm(self):
        """Inspector data is correctly encoded for a simple Ising triangle problem given as a BQM."""

        problem = (self.bqm_embedded.linear, self.bqm_embedded.quadratic)

        # sample
        with BrickedClient() as client:
            solver = client.get_solver()
            response = solver.sample_ising(*problem, **self.params)
            response.wait()

        # convert
        data = from_qmi_response(problem, response, params=self.params)

        # validate data encoding
        self.verify_data_encoding(problem=problem, response=response,
                                  solver=solver, params=self.params, data=data)

    @rec.use_cassette('triangle-qubo.yaml')
    def test_from_qmi_response__qubo(self):
        """Inspector data is correctly encoded for a simple QUBO triangle problem."""

        # vars = (0, 1, 12)
        # h = {}, J = {(0, 1): 1, (0, 12): 1, (1, 12): 1}
        problem = {
            (0, 1): 1, (0, 12): 1, (1, 12): 1
        }

        # sample
        with BrickedClient() as client:
            solver = client.get_solver()
            response = solver.sample_qubo(problem, **self.params)
            response.wait()

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
            solver = client.get_solver()
            response = solver.sample_ising(*problem, **self.params)
            response.wait()

        # convert
        data = from_qmi_response(problem, response, params=self.params)

        # validate data encoding
        self.verify_data_encoding(problem=problem, response=response,
                                  solver=solver, params=self.params, data=data)

    @rec.use_cassette('single-qubit-ising.yaml')
    def test_from_qmi_response__single_qubit(self):
        """Problem/solutions are correctly encoded for single-qubit problems."""

        problem = ({0: 1}, {})

        # sample
        with BrickedClient() as client:
            solver = client.get_solver()
            response = solver.sample_ising(*problem, **self.params)
            response.wait()

        # convert
        data = from_qmi_response(problem, response, params=self.params)

        # validate data encoding
        self.verify_data_encoding(problem=problem, response=response,
                                  solver=solver, params=self.params, data=data)

    @rec.use_cassette('triangle-ising.yaml')
    def test_from_qmi_response__problem_encoding(self):
        """Problem data is serialized even when it uses non-standard types (like numpy.int64)."""

        # `self.problem` == (
        #   {0: 0.0, 1: 0.0, 12: 0.0},
        #   {(0, 1): 1, (0, 12): 1, (1, 12): 1}
        # )
        h = {
            0: numpy.int64(0),
            1: numpy.double(0),
            12: numpy.int8(0),
        }
        J = {
            (0, 1): numpy.float16(1),
            (0, 12): Decimal('1'),
            (1, 12): Fraction(2, 2),
        }
        problem = (h, J)

        # sample
        with BrickedClient() as client:
            solver = client.get_solver()
            response = solver.sample_ising(*problem, **self.params)
            response.wait()

        # convert
        data = from_qmi_response(problem, response, params=self.params)

        # validate data encoding
        self.verify_data_encoding(problem=problem, response=response,
                                  solver=solver, params=self.params, data=data)

    @rec.use_cassette('triangle-ising.yaml')
    def _test_from_bqm_response(self, bqm):
        # sample
        with BrickedClient() as client:
            solver = client.get_solver()
            response = solver.sample_ising(*self.problem, **self.params)
            response.wait()

            # induce sampleset production in response, to test serialization of
            # sampleset-provided data, like `num_occurrences` (an numpy.ndarray)
            # NOTE: `dwave.cloud.computation.Future.num_occurrences` et al. will
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

    @rec.use_cassette('triangle-ising.yaml')
    def _test_from_bqm_sampleset(self, bqm):
        # sample
        qpu = DWaveSampler(solver=dict(topology__type='zephyr'))
        sampler = FixedEmbeddingComposite(qpu, self.embedding)
        sampleset = sampler.sample(
            bqm, return_embedding=True, chain_strength=self.chain_strength,
            **self.params)

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
            solver = client.get_solver()
            response = solver.sample_ising(*self.problem, **self.params)

            # resolve it before we mangle with it
            response.result()

        # change solver to unstructured to test solver validation
        response.solver = BQMSolver(client=None, data=hybrid_bqm_solver_data())

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
        qpu = DWaveSampler(solver=dict(topology__type='zephyr'))
        sampler = FixedEmbeddingComposite(qpu, self.embedding)
        sampleset = sampler.sample(self.bqm, return_embedding=True, **self.params)

        # resolve it before we mangle with it
        sampleset.info['problem_id']
        # change solver to unstructured to test solver validation
        sampler.child.solver = BQMSolver(client=None, data=hybrid_bqm_solver_data())

        # ensure `from_bqm_sampleset` adapter fails on unstructured solver
        with self.assertRaises(TypeError):
            from_bqm_sampleset(self.bqm, sampleset, sampler, params=self.params)

    @rec.use_cassette('triangle-ising.yaml')
    def test_solver_graph_validation(self):
        """All data adapters should fail on non-Chimera/Pegasus solvers."""

        # sample
        with BrickedClient() as client:
            solver = client.get_solver()
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
        qpu = DWaveSampler(solver=dict(topology__type='zephyr'))
        sampler = FixedEmbeddingComposite(qpu, self.embedding)
        sampleset = sampler.sample(self.bqm, return_embedding=True, **self.params)

        # resolve it before we mangle with it
        sampleset.info['problem_id']
        # change solver topology to non-chimera/pegasus to test solver validation
        sampler.child.solver.properties['topology']['type'] = 'unknown'

        # ensure `from_bqm_sampleset` adapter fails on unstructured solver
        with self.assertRaises(TypeError):
            from_bqm_sampleset(self.bqm, sampleset, sampler, params=self.params)

    @rec.use_cassette('triangle-ising-labelled.yaml')
    def test_problem_label_in_response(self):
        """All data adapters should propagate problem label."""

        # sample ising -> response
        with BrickedClient() as client:
            solver = client.get_solver()
            response = solver.sample_ising(*self.problem, label=self.label, **self.params)
            response.wait()

        # ensure `from_qmi_response` adapter propagates label
        data = from_qmi_response(self.problem, response, params=self.params)
        self.assertEqual(data['details']['label'], self.label)

        # ensure `from_bqm_response` adapter propagates label
        data = from_bqm_response(self.bqm, self.embedding_context, response, params=self.params)
        self.assertEqual(data['details']['label'], self.label)

    @rec.use_cassette('triangle-ising-labelled.yaml')
    def test_problem_label_in_sampleset(self):
        """All data adapters should propagate problem label."""

        # sample bqm -> sampleset
        qpu = DWaveSampler(solver=dict(topology__type='zephyr'))
        sampler = FixedEmbeddingComposite(qpu, self.embedding)
        sampleset = sampler.sample(self.bqm, label=self.label, **self.params)

        # ensure `from_bqm_sampleset` adapter propagates label
        data = from_bqm_sampleset(self.bqm, sampleset, sampler, params=self.params)
        self.assertEqual(data['details']['label'], self.label)

    @rec.use_cassette('triangle-ising.yaml')
    def test_implicit_solver_topology(self):
        """All data adapters should work on Chimera-implied solvers."""

        # sample
        with BrickedClient() as client:
            solver = client.get_solver()
            response = solver.sample_ising(*self.problem, **self.params)
            response.wait()

        # simulate old solver, without explicit topology property
        del response.solver.properties['topology']

        # convert and validate
        data = from_qmi_response(self.problem, response, params=self.params)
        self.verify_data_encoding(problem=self.problem, response=response,
                                  solver=solver, params=self.params, data=data)

        # in addition to `topology` missing, remove "structure", so Chimera
        # can't be implied
        del solver.properties['couplers']

        # ensure `from_qmi_response` adapter fails on unstructured old solver
        with self.assertRaises(TypeError):
            from_qmi_response(self.problem, response, params=self.params)

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
