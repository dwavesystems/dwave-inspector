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

__all__ = [
    'itemsgetter',
]

logger = logging.getLogger(__name__)


def itemsgetter(*items):
    """Variant of :func:`operator.itemgetter` that returns a callable that
    always returns a tuple, even when called with one argument. This is to make
    the result type consistent, regardless of input.
    """

    if len(items) == 1:
        item = items[0]
        def f(obj):
            return (obj[item], )

    else:
        f = operator.itemgetter(*items)

    return f
