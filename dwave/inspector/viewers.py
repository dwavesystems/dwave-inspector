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
import functools
import webbrowser
import operator
from pkg_resources import iter_entry_points

logger = logging.getLogger(__name__)


def annotated(**kwargs):
    """Decorator for annotating function objects with **kwargs attributes.

    Args:
        **kwargs (dict):
            Map of attribute values to names.

    Example:
        Decorate `f` with `priority=10`::

            @annotated(priority=10)
            def f():
                pass

            assert f.priority == 10

    """

    def _decorator(f):
        for key, val in kwargs.items():
            setattr(f, key, val)
        return f

    return _decorator


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

    viewers = [ep.load() for ep in iter_entry_points('inspectorapp_viewers')]
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

    return False
