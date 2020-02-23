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
import enum

from dwave.cloud.utils import set_loglevel

from dwave.inspector.server import app_server
from dwave.inspector.adapters import (
    from_qmi_response, from_bqm_response, from_bqm_sampleset, from_objects,
    enable_data_capture)
from dwave.inspector.storage import push_inspector_data
from dwave.inspector.viewers import view
from dwave.inspector.package_info import __version__, __author__, __description__

# expose the root logger to simplify access
logger = logging.getLogger(__name__)

def _configure_logging(logger, loglevel):
    """Configure `logger` root logger."""
    # TODO: move to dwave "common utils" module

    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(threadName)s %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    set_loglevel(logger, loglevel)

# configure root logger and apply `DWAVE_INSPECTOR_LOG_LEVEL`
_configure_logging(logger, os.getenv('DWAVE_INSPECTOR_LOG_LEVEL'))


# enable inspector data capture on import!
enable_data_capture()


class Block(enum.Enum):
    NEVER = 'never'
    ONCE = 'once'
    FOREVER = 'forever'


def open_problem(problem_id, block=Block.ONCE):
    """Open problem_id from storage in the Inspector web app.

    Args:
        problem_id (str):
            Submitted problem id, as returned by SAPI.

        block (:class:`Block`/str/bool, optional, default=:obj:`Block.ONCE`):
            Blocking behavior after opening up the web browser preview. In
            between the obvious edge cases (:obj:`Block.NEVER`/'never'/:obj:`False`
            and :obj:`Block.FOREVER`/'forever'/:obj:`True`), a value of
            :obj:`Block.ONCE`/'once' will block the return until the problem has
            been loaded from the inspector web server exactly once.

    """
    # accept string name for `block`
    if isinstance(block, str):
        block = Block(block.lower())

    app_server.ensure_started()
    url = app_server.get_inspect_url(problem_id)

    # open url and block if requested
    view(url)

    if block is Block.ONCE:
        app_server.wait_problem_accessed(problem_id)
    elif block is Block.FOREVER or block is True:
        app_server.wait_shutdown()

    return url


def show_qmi(problem, response, embedding_context=None, warnings=None, params=None):
    data = from_qmi_response(problem=problem, response=response,
                             embedding_context=embedding_context,
                             warnings=warnings, params=params)
    id_ = push_inspector_data(data)
    return open_problem(id_)


def show_bqm_response(bqm, embedding_context, response, warnings=None, params=None):
    data = from_bqm_response(bqm=bqm, embedding_context=embedding_context,
                             response=response, warnings=warnings, params=params)
    id_ = push_inspector_data(data)
    return open_problem(id_)


def show_bqm_sampleset(bqm, sampleset, sampler, embedding_context=None,
                       warnings=None, params=None):
    data = from_bqm_sampleset(bqm=bqm, sampleset=sampleset, sampler=sampler,
                              embedding_context=embedding_context,
                              warnings=warnings, params=params)
    id_ = push_inspector_data(data)
    return open_problem(id_)


def show(*args, **kwargs):
    """Auto-detect the optimal `show_*` method based on arguments provided and
    forward the call.

    Note:
        Low-level data capture is enabled on `dwave.inspector` import. Data
        captured includes the full QMI, QPU response, embedding context,
        warnings, and sampling parameters.

        If the data capture is enabled prior to embedding/sampling, the only
        necessary argument to provide to `show()` is a response / problem id
        (for QMI inspection) or a sampleset (for logical problem + QMI
        inspection).

        The alternative (late import) requires all relevant data to be
        explicitly provided to `show()`.

    Examples:

        # QMI-only viz (no logical problem)
        show((h, J), response)
        show(Q, response)
        show(response)
        show('69ace80c-d3b1-448a-a028-b51b94f4a49d')

        # QMI + explicit embedding (-> no warnings! fix!)
        show((h, J), response, dict(embedding=embedding, chain_strength=5))

        # embedding and warnings read from the sampleset
        show(bqm, sampleset)

        # embedding/warnings/problem_id read from sampleset, logical problem reconstructed
        show(sampleset)

    """
    block = kwargs.pop('block', Block.ONCE)
    data = from_objects(*args, **kwargs)
    id_ = push_inspector_data(data)
    return open_problem(id_, block=block)
