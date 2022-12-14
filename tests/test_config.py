# Copyright 2022 D-Wave Systems Inc.
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

from dwave.cloud.testing import isolated_environ

from dwave.inspector.config import config


class TestConfig(unittest.TestCase):

    @isolated_environ(empty=True)
    def test_defaults(self):
        self.assertIsNone(config.log_level)
        self.assertEqual(config.host, '127.0.0.1')
        self.assertEqual(config.base_port, 18000)

    @isolated_environ(empty=True, add=dict(
        DWAVE_INSPECTOR_LOG_LEVEL='debug',
        DWAVE_INSPECTOR_HOST='localhost',
        DWAVE_INSPECTOR_BASE_PORT='9000',
        DWAVE_INSPECTOR_JUPYTER_SERVER_PROXY_EXTERNAL_URL='url'
    ))
    def test_env(self):
        self.assertEqual(config.log_level, 'debug')
        self.assertEqual(config.host, 'localhost')
        self.assertEqual(config.base_port, 9000)
        self.assertEqual(config.jupyter_server_proxy_external_url, 'url')
