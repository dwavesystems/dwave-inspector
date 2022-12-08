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

import logging
import operator
import webbrowser
from pkg_resources import iter_entry_points

from dwave.inspector.package_info import entry_point_group
from dwave.inspector.utils import annotated

logger = logging.getLogger(__name__)


@annotated(priority=1000)
def jupyter_inline(rich_url):
    """Hijack viewers (use high priority) and prevent browser popping up when
    running in interactive (GUI) Jupyter session. That way only the inline
    Inspector is shown.
    """
    # note: `get_ipython` is available without import since ipython 5.1
    # (and it's fine to fail here, since the next viewer is attempted in that case)
    ipython = get_ipython()
    logger.debug('Running inside ipython: %r', ipython)
    if 'ZMQInteractiveShell' not in type(ipython).__name__:
        raise ValueError('non-gui interactive shell')

    # render URL/IFrame inline in jupyter notebook, or fail trying
    # note: since ipython 5.4/6.1 (May 2017) `display` is available without import
    display(rich_url)

    # don't block if gui interactive shell is used
    return False


@annotated(priority=0)
def webbrowser_tab(url):
    return webbrowser.open_new_tab(url)


@annotated(priority=-10)
def webbrowser_window(url):
    return webbrowser.open_new(url)


def prioritized_viewers():
    """Return all registered InspectorApp viewers order by descending
    priority.
    """

    viewers = [ep.load() for ep in iter_entry_points(entry_point_group['viewers'])]
    return sorted(viewers, key=operator.attrgetter('priority'), reverse=True)


def view(url):
    """Open URL with the highest priority viewer that accepts it."""

    for viewer in prioritized_viewers():
        try:
            logger.debug('Trying to open the webapp URL %r with %r',
                         url, viewer.__name__)
            return viewer(url)

        except Exception as exc:
            logger.info('Opening the webapp URL with %r failed with %r',
                        viewer.__name__, exc)

    # caller should not block on server access, since no viewers opened
    return False
