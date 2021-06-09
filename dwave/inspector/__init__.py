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

import os
import enum
import logging

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
    """Flow-control settings for scripts.

    An enum with values: ``NEVER``, ``ONCE``, ``FOREVER``. The default setting of
    ``once`` (``dwave.inspector.Block.ONCE``) blocks until your problem
    is loaded from the inspector web server.

    Examples:
        This example does not block while the problem is loaded.

        >>> dwave.inspector.show(response, block='never')   # doctest: +SKIP

    """
    NEVER = 'never'
    ONCE = 'once'
    FOREVER = 'forever'


def open_problem(problem_id, block=Block.ONCE, timeout=None):
    """Open the problem inspector for the specified problem.

    Args:
        problem_id (str):
            Submitted problem identity, as returned by SAPI.

        block (:class:`Block`/str/bool, optional, default: :obj:`Block.ONCE`):
            Blocking behavior after opening up the web browser preview as set
            by :class:`Block` value.

        timeout (float):
            Blocking behavior timeout in seconds.

    """
    # accept string name for `block`
    if isinstance(block, str):
        block = Block(block.lower())

    app_server.ensure_started()
    url = app_server.get_inspect_url(problem_id)

    # open url and block if requested
    view(url)

    if block is Block.ONCE:
        app_server.wait_problem_accessed(problem_id, timeout=timeout)
    elif block is Block.FOREVER or block is True:
        app_server.wait_shutdown(timeout=timeout)

    return url


def show_qmi(problem, response, embedding_context=None, warnings=None, params=None):
    """
    Visualize a quantum machine instruction (QMI).
    """
    data = from_qmi_response(problem=problem, response=response,
                             embedding_context=embedding_context,
                             warnings=warnings, params=params)
    id_ = push_inspector_data(data)
    return open_problem(id_)


def show_bqm_response(bqm, embedding_context, response, warnings=None, params=None):
    """
    Visualize a quantum machine instruction (QMI) response and binary quadratic model.
    """
    data = from_bqm_response(bqm=bqm, embedding_context=embedding_context,
                             response=response, warnings=warnings, params=params)
    id_ = push_inspector_data(data)
    return open_problem(id_)


def show_bqm_sampleset(bqm, sampleset, sampler, embedding_context=None,
                       warnings=None, params=None):
    """
    Visualize a returned sampleset and binary quadratic model.
    """
    data = from_bqm_sampleset(bqm=bqm, sampleset=sampleset, sampler=sampler,
                              embedding_context=embedding_context,
                              warnings=warnings, params=params)
    id_ = push_inspector_data(data)
    return open_problem(id_)


def show(*args, **kwargs):
    """Auto-detect and forward to the ``show_*`` optimal for the specified
    arguments.

    For description of accepted arguments, see of :func:`.show_qmi`,
    :func:`.show_bqm_response`, or :func:`.show_bqm_sampleset`.

    Note:
        Low-level data capture is enabled on `dwave.inspector` import. Data
        captured includes the full quantum machine instruction (QMI), QPU
        response, embedding context, warnings, and sampling parameters.

        If data capture is enabled prior to embedding/sampling, you need
        provide to :func:`~dwave.inspector.show` only a response or problem ID
        for QMI inspection or a :class:`~dimod.SampleSet` for logical problem
        and QMI inspection.

        If data capture is not enabled prior to embedding/sampling, provide
        all relevant data explicitly to :func:`~dwave.inspector.show`.

    Examples:

        This example shows ways to visualize just a QMI (not a logical problem)::

            show(response)
            show((h, J), response)
            show(Q, response)
            show('69ace80c-d3b1-448a-a028-b51b94f4a49d')

        This example visualizes a QMI and explicit embedding::

            show((h, J), response, dict(embedding=embedding, chain_strength=5))

        This example shows embedding and warnings read from the sampleset::

            show(bqm, sampleset)

        This example shows embedding and warnings read from a :class:`~dimod.SampleSet`,
        from which the logical problem is reconstructed::

            show(sampleset)

    """
    block = kwargs.pop('block', Block.ONCE)
    timeout = kwargs.pop('timeout', None)
    data = from_objects(*args, **kwargs)
    id_ = push_inspector_data(data)
    return open_problem(id_, block=block, timeout=timeout)
