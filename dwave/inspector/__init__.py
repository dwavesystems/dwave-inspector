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
from dwave.inspector.adapters import from_qmi_response, from_bqm_response
from dwave.inspector.storage import push_problem


def open_problem(problem_id):
    """Open problem_id from storage in the Inspector web app."""
    app_server.ensure_started()
    url = "http://localhost:8000/?testId={}".format(problem_id)
    webbrowser.open_new_tab(url)
    return url


def show_qmi(problem, response, embedding=None):
    problem = from_qmi_response(problem, response, embedding)
    id_ = push_problem(problem)
    return open_problem(id_)


def show_bqm_response(bqm, embedding, response, warnings=None):
    problem = from_bqm_response(bqm, embedding, response, warnings)
    id_ = push_problem(problem)
    return open_problem(id_)
