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

from time import perf_counter

import vcr
from packaging.specifiers import SpecifierSet

from dwave.cloud import Client
from dwave.cloud.package_info import __version__ as cc_version


# NOTE: copied from dwave-hybrid
# TODO: Remove/replace when dwave.common utility package is created
class RunTimeAssertionMixin:
    """unittest.TestCase mixin that adds min/max/range run-time assert."""

    class assertRuntimeWithin:

        def __init__(self, low, high):
            """Min/max runtime in milliseconds."""
            self.limits = (low, high)

        def __enter__(self):
            self.tick = perf_counter()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.dt = (perf_counter() - self.tick) * 1000.0
            self.test()

        def test(self):
            low, high = self.limits
            if low is not None and self.dt < low:
                raise AssertionError("Min runtime unreached: %g ms < %g ms" % (self.dt, low))
            if high is not None and self.dt > high:
                raise AssertionError("Max runtime exceeded: %g ms > %g ms" % (self.dt, high))

    class assertMinRuntime(assertRuntimeWithin):

        def __init__(self, t):
            """Min runtime in milliseconds."""
            self.limits = (t, None)

    class assertMaxRuntime(assertRuntimeWithin):

        def __init__(self, t):
            """Max runtime in milliseconds."""
            self.limits = (None, t)


# client factory that turns off on-disk caching and removes the token requirement
def BrickedClient(**kwargs):
    # currently, this is the only way to skip on-disk caching (we do not want
    # the client to use the existing cache during tests -- that way we can control
    # the SAPI endpoint returned from the Metadata API)
    # TODO: replace with cache-control when dwave-cloud-client#503 is implemented.
    if hasattr(Client, '_fetch_available_regions'):
        Client._fetch_available_regions._cached.cache = {}
    else:
        # in `dwave-cloud-client < 0.9.2` there's no cache we need to mock
        pass

    # we can use a fake token because requests are replayed anyway
    client = Client(token='fake', **kwargs)

    return client


# SAPI responses changed in v3 (with cloud-client 0.14.0)
if cc_version in SpecifierSet('>=0.14.0rc', prereleases=True):
    sapi_version = 'v3'
else:
    sapi_version = 'v2'

sapi_vcr = vcr.VCR(
    serializer='yaml',
    cassette_library_dir=f'tests/fixtures/cassettes/{sapi_version}',
    record_mode='none',
    match_on=['uri', 'method'],
    filter_headers=['x-auth-token'],
    filter_query_parameters=['timeout'],
    ignore_localhost=True,
)
