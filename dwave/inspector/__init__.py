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

import os
import logging
import webbrowser

from dwave.cloud.utils import set_loglevel

from dwave.inspector.server import app_server
from dwave.inspector.adapters import (
    from_qmi_response, from_bqm_response, from_bqm_sampleset, from_objects)
from dwave.inspector.storage import push_problem


def _configure_logging(loglevel):
    """Configure `dwave.inspector` root logger."""
    # TODO: move to dwave "common utils" module

    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(threadName)s %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.addHandler(handler)

    set_loglevel(logger, loglevel)

# configure root logger and apply `DWAVE_INSPECTOR_LOG_LEVEL`
_configure_logging(os.getenv('DWAVE_INSPECTOR_LOG_LEVEL'))


def open_problem(problem_id):
    """Open problem_id from storage in the Inspector web app."""
    app_server.ensure_started()
    url = "http://localhost:8000/?testId={}".format(problem_id)
    webbrowser.open_new_tab(url)
    return url


def show_qmi(problem, response, embedding=None, warnings=None):
    problem = from_qmi_response(problem, response, embedding)
    id_ = push_problem(problem)
    return open_problem(id_)


def show_bqm_response(bqm, embedding, response, warnings=None):
    problem = from_bqm_response(bqm, embedding, response, warnings)
    id_ = push_problem(problem)
    return open_problem(id_)


def show_bqm_sampleset(bqm, sampleset, sampler, embedding=None, warnings=None):
    problem = from_bqm_sampleset(bqm, sampleset, sampler, embedding, warnings)
    id_ = push_problem(problem)
    return open_problem(id_)


def show(*args, **kwargs):
    """Auto-detect the optimal `show_*` method based on arguments provided and
    forward the call.
    """
    problem = from_objects(*args, **kwargs)
    id_ = push_problem(problem)
    return open_problem(id_)
