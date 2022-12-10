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

"""D-Wave Inspector configuration as loaded from the environment variables."""

import os


# note: for now, we only support environment, but in future we might add support
# for config files

class Config:

    @staticmethod
    def _env(key, default=None):
        return os.getenv(key, default)

    @property
    def log_level(self):
        return self._env('DWAVE_INSPECTOR_LOG_LEVEL')

    @property
    def host(self):
        return self._env('DWAVE_INSPECTOR_HOST', '127.0.0.1')

    @property
    def base_port(self):
        return int(self._env('DWAVE_INSPECTOR_BASE_PORT', 18000))

    @property
    def jupyter_server_proxy_external_url(self):
        return self._env('DWAVE_INSPECTOR_JUPYTER_SERVER_PROXY_EXTERNAL_URL')


config = Config()
