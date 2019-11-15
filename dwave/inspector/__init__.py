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

import webbrowser

from dwave.inspector.server import app_server
from dwave.inspector.adapters import from_objects
from dwave.inspector.storage import problem_store


def show(bqm=None, embedding=None, response=None, warnings=None):
    problem = from_objects(bqm, embedding, response, warnings)
    id_ = problem['details']['id']
    problem_store[id_] = problem

    app_server.ensure_started()

    webbrowser.open_new_tab("http://localhost:8000/?testId={}".format(id_))
