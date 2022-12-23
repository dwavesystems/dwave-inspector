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

__all__ = [
    '__package_name__', '__title__', '__version__',
    '__author__', '__author_email__', '__description__',
    '__url__', '__license__', '__copyright__'
]

__package_name__ = 'dwave-inspector'
__title__ = 'D-Wave Problem Inspector'
__version__ = '0.4.0'
__author__ = 'D-Wave Systems Inc.'
__author_email__ = 'radomir@dwavesys.com'
__description__ = 'D-Wave Problem Inspector tool'
__url__ = 'https://github.com/dwavesystems/dwave-inspector'
__license__ = 'Apache 2.0'
__copyright__ = '2019, D-Wave Systems Inc.'


# register the non-open-source packages required for dwave-inspector to work
contrib = [{
    'name': 'inspector',
    'title': 'D-Wave Problem Inspector',
    'description': 'This tool visualizes problems submitted to the quantum computer and the results returned.',
    'license': {
        'name': 'D-Wave EULA',
        'url': 'https://docs.ocean.dwavesys.com/eula',
    },
    'requirements': [
        'dwave-inspectorapp==0.3.1',
    ]
}]


# all entry point groups in one place
entry_point_group = {
    'viewers': 'inspectorapp_viewers',
    'proxies': 'inspectorapp_proxies',
    'contrib': 'dwave_contrib',
}
