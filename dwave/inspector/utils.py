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

import re
import sys
import logging
import operator
import functools

from typing import Sequence, Callable, Optional, Union, Dict
from urllib.parse import urlparse, ParseResult

try:
    from importlib.metadata import EntryPoint, DistributionFinder, Distribution
except ImportError: # pragma: no cover
    from importlib_metadata import EntryPoint, DistributionFinder, Distribution

import flask
import numpy

__all__ = [
    'itemsgetter', 'annotated', 'NumpyJSONProvider', 'patch_entry_points',
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


# adapted from dwave-hybrid utils
# (https://github.com/dwavesystems/dwave-hybrid/blob/b9025b5bb3d88dce98ec70e28cfdb25400a10e4a/hybrid/utils.py#L43-L61)
# to provide numpy types serialization in flask 2.2+ (see https://github.com/pallets/flask/pull/4692)
class NumpyJSONProvider(flask.json.provider.DefaultJSONProvider):
    """JSON encoder for numpy types.

    Supported types:
     - basic numeric types: booleans, integers, floats
     - arrays: ndarray, recarray
    """

    @staticmethod
    def default(obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.bool_):
            return bool(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()

        return super().default(obj)


# TODO: move to `dwave.common.testing` or similar
class patch_entry_points:
    """Patch entry points via an in-memory-installed distribution that provides
    the new (temporary) entry points.
    """

    class InMemoryDistribution(Distribution):
        def __init__(self, group: str, eps: Sequence[Callable]):
            self._group = group
            self._eps = eps

        def read_text(self, filename):
            if filename == 'METADATA':
                return (
                    "Name: in-memory-distribution\n"
                    "Version: 0.0.0"
                )

        def locate_file(self, path):
            pass

        @property
        def entry_points(self):
            eps = [EntryPoint(name=ep.__name__,
                              value=f'{ep.__module__}:{ep.__name__}',
                              group=self._group)
                   for ep in self._eps]
            return eps

    class InMemoryDistributionFinder(DistributionFinder):
        def __init__(self, dists: Optional[Sequence[Distribution]] = None):
            self.dists = dists

        def find_distributions(self, context):
            yield from self.dists

    def __init__(self, group: str, eps: Sequence):
        """Hook functions listed in ``eps`` to ``group`` entry point."""
        self.dist = self.InMemoryDistribution(group=group, eps=eps)

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            finder = self.InMemoryDistributionFinder((self.dist, ))
            sys.meta_path.append(finder)

            try:
                return fn(*args, **kwargs)
            finally:
                sys.meta_path.remove(finder)

        return wrapper


def update_url_from(url: Union[str, ParseResult],
                    patch: Union[str, ParseResult],
                    **merge_op: Optional[Dict[str, Callable[[ParseResult, ParseResult], str]]]) -> str:
    """Update ``url`` with components from ``patch`` using ``merge_op`` functions.

    Inputs can be given in string form, or as :class:`~urllib.parse.ParseResult`
    (namedtuple). Output type is always string.

    Args:
        url:
            Input URL.
        patch:
            URL patch.
        **merge_op:
            URL component merge operator, one for each component: ``scheme``,
            ``netloc``, ``path``, ``params``, ``query``, and ``fragment``.
            If unspecified, a non-null component from patch will overwrite the
            component from url.

    Returns:
        Updated ``url`` with components from ``patch`` using ops from ``merge_op``.

    Example::

        url = update_url_from(
            'http://localhost:8000/notebook',
            'https://example.com/prefix?username=lisa',
            path=lambda src, dst: f'{dst.path}{src.path}')

        assert(url, 'https://example.com/prefix/notebook?usernmae=lisa')

    """

    # handle schemeless urls -> assume http
    if not re.match("^\w+://", url):
        url = f"http://{url}"

    # deconstruct source and patch urls
    if not isinstance(url, ParseResult):
        url = urlparse(url)
    if not isinstance(patch, ParseResult):
        patch = urlparse(patch)

    default_for = \
        lambda field: \
            lambda url, patch: getattr(patch, field, '') or getattr(url, field)

    res = {field: merge_op.get(field, default_for(field))(url, patch) for field in url._fields}
    return ParseResult(**res).geturl()
