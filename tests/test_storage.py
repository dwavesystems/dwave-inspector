# Copyright 2020 D-Wave Systems Inc.
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
from unittest import mock

from dwave.inspector.storage import push_inspector_data, get_solver_data

from tests import BrickedClient, sapi_vcr as rec


@mock.patch('dwave.system.samplers.dwave_sampler.Client.from_config', BrickedClient)
class TestStorage(unittest.TestCase):

    @rec.use_cassette('triangle-ising.yaml')
    def test_solver_modernization(self):
        """Solver's data is "modernized", i.e. missing props are added."""

        # get real solver
        with BrickedClient() as client:
            solver = client.get_solver()

        # cripple it
        del solver.properties['topology']

        # mock minimal data adapter response
        inspector_data = {'rel': {'solver': solver}, 'details': {'id': 'mock-id'}}

        # store it
        push_inspector_data(inspector_data)

        # get solver data
        solver_data = get_solver_data(solver.id, update_inplace=False)

        # verify missing data is recreated
        self.assertIn('topology', solver_data['properties'])
        self.assertEqual(solver_data['properties']['topology']['type'], 'chimera')

        # verify the original solver data is intact
        self.assertNotIn('topology', solver.properties)
