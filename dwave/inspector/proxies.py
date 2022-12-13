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

"""
General future-proof proxy function signature::

    @annotated(priority: int, **attrs)
    def proxy(**kwargs):
        # return result, side-effects possible
        # yield to next rewriter by raising an exception

Concrete URL rewriter function signature::

    @annotated(priority: int, url_rewriter=True)
    def url_rewriter(url, **kwargs):
        # return new url, no side-effects
        # yield to next rewriter by raising an exception

"""

import logging
import operator
from typing import List, Callable
from urllib.parse import urljoin

try:
    from importlib_metadata import entry_points
except ImportError:  # noqa
    # proper native support in py310+
    from importlib.metadata import entry_points

from dwave.inspector.package_info import entry_point_group
from dwave.inspector.utils import annotated, update_url_from
from dwave.inspector.config import config

logger = logging.getLogger(__name__)


@annotated(priority=-10, url_rewriter=True)
def jupyter_server_proxy(url, **kwargs):
    # note: jupyter server proxy has to be installed and configured
    if not config.jupyter_server_proxy_external_url:
        raise ValueError('jupyter-server-proxy external URL not configured')

    return update_url_from(
        url, config.jupyter_server_proxy_external_url,
        path=lambda local, ext: urljoin(ext.path, f"{local.path}/proxy/{local.port}/".lstrip('/')))


def prioritized_url_rewriters() -> List[Callable]:
    """Return all registered URL rewriters, ordered by descending priority."""

    proxies = [ep.load() for ep in entry_points().select(group=entry_point_group['proxies'])]
    rewriters = filter(lambda f: getattr(f, 'url_rewriter', None), proxies)
    return sorted(rewriters, key=operator.attrgetter('priority'), reverse=True)


def rewrite_url(url: str, **kwargs) -> str:
    """Rewrite the internal `url` with the highest priority url rewriter that
    accepts it.
    """

    for rewriter in prioritized_url_rewriters():
        try:
            logger.debug('Invoking URL rewriter %s(%r, %r)',
                         rewriter.__name__, url, kwargs)
            return rewriter(url, **kwargs)

        except Exception as exc:
            logger.info('URL rewriter %r declined to rewrite with %r',
                        rewriter.__name__, exc)

    # fallback to no rewrite
    return url
