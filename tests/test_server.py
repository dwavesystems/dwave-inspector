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
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import vcr
import requests

from dwave.cloud import Client

from dwave.inspector.server import app_server
from dwave.inspector import show, Block

from tests import RunTimeAssertionMixin


rec = vcr.VCR(
    serializer='yaml',
    cassette_library_dir='tests/fixtures/cassettes',
    record_mode='none',
    match_on=['uri', 'method'],
    filter_headers=['x-auth-token'],
    ignore_localhost=True,
)

BrickedClient = partial(Client, token='fake')


class TestServerRuns(unittest.TestCase):

    def test_lazy_start(self):
        self.assertIsNone(getattr(app_server, '_server', None))
        self.assertIsNotNone(app_server.server)

    def test_smoke(self):
        self.assertTrue(app_server.ensure_started())
        # NOTE: we shouldn't test app_server.stop() if other tests need to run
        # the server again, since currently our app server can be started ONLY
        # ONCE per program run.


@unittest.mock.patch('dwave.system.samplers.dwave_sampler.Client.from_config', BrickedClient)
@unittest.mock.patch('dwave.inspector.view', lambda url: None)
class TestServerWorks(unittest.TestCase, RunTimeAssertionMixin):

    @rec.use_cassette('triangle-ising.yaml')
    @classmethod
    def setUpClass(cls):
        # sample
        with BrickedClient() as client:
            solver = client.get_solver(qpu=True)
            cls.response = solver.sample_ising({}, {(0, 4): 1, (0, 5): 1, (4, 1): 1, (1, 5): -1})
            cls.problem_id = cls.response.wait_id()

    @staticmethod
    def notify_problem_loaded(problem_id):
        url = app_server.get_callback_url(problem_id)
        return requests.get(url).json()

    def test_show_no_block(self):
        # exclude potential server start-up time from timing tests below
        app_server.ensure_started()

        # show shouldn't block regardless of problem inspector opening or not
        with self.assertMaxRuntime(2000):
            show(self.response, block=Block.NEVER)

    def test_show_block_once(self):
        # exclude potential server start-up time from timing tests below
        app_server.ensure_started()

        # show should block until first access
        with ThreadPoolExecutor(max_workers=1) as executor:
            with self.assertMaxRuntime(2000):
                # notify the inspector server the `problem_id` has been "viewed" by
                # our mock browser/viewer (async)
                fut = executor.submit(self.notify_problem_loaded, problem_id=self.problem_id)
                show(self.response, block=Block.ONCE)
                fut.result()

    def test_show_block_timeout(self):
        # exclude potential server start-up time from timing tests below
        app_server.ensure_started()

        # show blocks until timeout
        with self.assertMaxRuntime(2000):
            show(self.response, block=Block.ONCE, timeout=1)

        with self.assertMaxRuntime(2000):
            show(self.response, block=True, timeout=1)
