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

import unittest
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import numpy as np
import requests

from dwave.system import DWaveSampler, FixedEmbeddingComposite

from dwave.inspector import show, Block
from dwave.inspector.server import app_server
from dwave.inspector.storage import problem_access_sem

from tests import RunTimeAssertionMixin, BrickedClient, sapi_vcr as rec


class FirstTestServerRuns(unittest.TestCase):

    def test_lazy_start(self):
        # note: this test has to run first
        self.assertIsNone(getattr(app_server, '_server', None))
        self.assertIsNotNone(app_server.server)

    def test_smoke(self):
        self.assertTrue(app_server.ensure_started())
        # NOTE: we shouldn't test app_server.stop() if other tests need to run
        # the server again, since currently our app server can be started ONLY
        # ONCE per program run.


@unittest.mock.patch('dwave.system.samplers.dwave_sampler.Client.from_config', BrickedClient)
class TestProblemOpen(unittest.TestCase, RunTimeAssertionMixin):

    @staticmethod
    def notify_problem_loaded(problem_id):
        url = app_server.get_callback_url(problem_id)
        return requests.get(url).json()

    @staticmethod
    def reset_problem_access_sem(problem_id):
        cnt = 0
        if problem_id not in problem_access_sem:
            return cnt
        while problem_access_sem[problem_id].acquire(blocking=False):
            cnt += 1
        return cnt

    @rec.use_cassette('triangle-ising.yaml')
    @classmethod
    def setUpClass(cls):
        # sample
        with BrickedClient() as client:
            solver = client.get_solver()
            cls.response = solver.sample_ising({}, {(0, 1): 1, (1, 12): 1, (12, 0): 1})
            cls.problem_id = cls.response.wait_id()
            cls.response.wait()

    def setUp(self):
        # exclude potential server start-up time from timing tests
        app_server.ensure_started()

        # reset inspector app access counter
        self.reset_problem_access_sem(self.problem_id)

    @unittest.mock.patch('dwave.inspector.view', lambda url: None)
    def test_show_no_block(self):
        # show shouldn't block regardless of problem inspector opening or not
        with self.assertMaxRuntime(2000):
            show(self.response, block=Block.NEVER)

    @unittest.mock.patch('dwave.inspector.view', lambda url: True)
    def test_show_block_once(self):
        # show should block until first access
        with ThreadPoolExecutor(max_workers=1) as executor:
            with self.assertMaxRuntime(2000):
                # notify the inspector server the `problem_id` has been "viewed" by
                # our mock browser/viewer (async)
                fut = executor.submit(self.notify_problem_loaded, problem_id=self.problem_id)
                show(self.response, block=Block.ONCE)
                fut.result()

    @unittest.mock.patch('dwave.inspector.view', lambda url: None)
    def test_show_block_timeout(self):
        # show blocks until timeout
        with self.assertRaises(TimeoutError):
            with self.assertMaxRuntime(2000):
                show(self.response, block=Block.ONCE, timeout=1)

        with self.assertRaises(TimeoutError):
            with self.assertMaxRuntime(2000):
                show(self.response, block=True, timeout=1)

    @unittest.mock.patch('dwave.inspector.view', lambda url: False)
    def test_show_blocking_ignored(self):
        # show shouldn't block regardless of problem inspector opening or not
        with self.assertMaxRuntime(2000):
            show(self.response, block=Block.ONCE)

    @unittest.mock.patch('dwave.inspector.view', lambda url: None)
    def test_api_access(self):
        # push data, for server to be able to fetch it
        show(self.response, block=Block.NEVER)

        # verify problem data access
        url = app_server.get_problem_url(self.problem_id)
        res = requests.get(url)
        self.assertEqual(res.status_code, 200)

        # verify solver data access
        url = app_server.get_solver_url(self.problem_id)
        res = requests.get(url)
        self.assertEqual(res.status_code, 200)

        # verify problem data access
        url = app_server.get_callback_url(self.problem_id)
        res = requests.get(url)
        self.assertEqual(res.status_code, 200)

    @unittest.mock.patch('dwave.inspector.view', lambda url: None)
    def test_redirect_root_to_last_problem(self):
        # push data, for server to be able to fetch it
        show(self.response, block=Block.NEVER)

        # verify base case: url with problemId
        url = app_server.get_problem_url(self.problem_id)
        res = requests.get(url)
        self.assertEqual(len(res.history), 0)   # no redirects
        self.assertEqual(res.status_code, 200)

        # verify redirect from root to last problem id
        base_url = urlsplit(url)._replace(path='/', query='', fragment='').geturl()
        res = requests.get(base_url)
        # one redirect
        self.assertEqual(len(res.history), 1)
        self.assertEqual(res.history[0].status_code, 302)
        # correct final location
        self.assertIn(self.problem_id, res.url)


@unittest.mock.patch('dwave.system.samplers.dwave_sampler.Client.from_config', BrickedClient)
class TestSerialization(unittest.TestCase):

    @rec.use_cassette('triangle-ising.yaml')
    @unittest.mock.patch('dwave.inspector.view', lambda url: None)
    def test_numpy_types_serialized(self):
        # model and embedding use variables with numpy types
        h = {}
        J = {
            (np.int8(0), np.int16(1)): np.int32(1),
            (np.int32(0), np.int64(12)): np.float32(1),
            (np.int64(1), np.int32(12)): np.float64(1),
        }
        embedding = {
            np.int8(0): [np.int8(0)],
            np.int32(1): [np.int32(1)],
            np.int64(12): [np.int64(12)],
        }

        # sample
        qpu = DWaveSampler(solver=dict(topology__type='zephyr'))
        sampler = FixedEmbeddingComposite(qpu, embedding)
        sampleset = sampler.sample_ising(h, J, return_embedding=True)
        problem_id = sampleset.info['problem_id']

        # push data, for server to be able to fetch it
        show(sampleset, block=Block.NEVER)

        # simulate problem data fetch from the inspectorapp
        url = app_server.get_problem_url(problem_id)
        res = requests.get(url)
        self.assertEqual(res.status_code, 200)
