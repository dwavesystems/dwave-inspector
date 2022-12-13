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

import uuid
import unittest

from dwave.cloud.testing import isolated_environ

from dwave.inspector.config import config
from dwave.inspector.utils import annotated, patch_entry_points
from dwave.inspector.proxies import prioritized_url_rewriters, rewrite_url
from dwave.inspector.package_info import entry_point_group


@annotated(priority=0, test_proxy=True, url_rewriter=True)
def proxy_nop(url, **kwargs):
    return url

@annotated(priority=1, test_proxy=True, proxy_server=True)
def proxy_server(**kwargs):
    return

@annotated(priority=2, test_proxy=True, url_rewriter=True)
def proxy_rewriter(url, **kwargs):
    return f'proxy://{url}'

@annotated(priority=3, test_proxy=True, url_rewriter=True)
def proxy_fail(url, **kwargs):
    raise ValueError


class TestProxies(unittest.TestCase):

    @patch_entry_points(group=entry_point_group['proxies'], eps=(proxy_nop, proxy_server, proxy_rewriter))
    def test_url_rewriters_prioritization(self):
        """URL rewriters are correctly filtered and ordered (by desc priority)."""

        # note: we only want to consider test proxies added
        pri = [v.priority for v in prioritized_url_rewriters()
               if getattr(v, 'test_proxy', False)]

        self.assertEqual(len(pri), 2)
        self.assertEqual(pri, sorted(pri, reverse=True))

    def test_default(self):
        """Default to original URL in case no rewriters registered."""

        url = str(uuid.uuid4())
        self.assertEqual(rewrite_url(url), url)

    @patch_entry_points(group=entry_point_group['proxies'], eps=(proxy_nop, proxy_server, proxy_rewriter))
    def test_rewrite(self):
        """URL rewritten."""

        url = str(uuid.uuid4())
        self.assertEqual(rewrite_url(url), f'proxy://{url}')

    @patch_entry_points(group=entry_point_group['proxies'], eps=(proxy_fail, proxy_nop, proxy_rewriter))
    def test_failure(self):
        """Next rewriter (by priority) is retried on failure."""

        url = str(uuid.uuid4())
        self.assertEqual(rewrite_url(url), f'proxy://{url}')

    def test_jupyter_server_proxy(self):
        """URL is rewritten properly with jupyter-server-proxy in play."""

        port = 18000
        local_url = f'http://localhost:{port}/?problemId=1'
        ext_base = 'https://example.com/jupyter/'

        with self.subTest('jupyter-server-proxy env var unset'):
            with isolated_environ(empty=True):
                self.assertEqual(rewrite_url(local_url), local_url)

        with self.subTest('jupyter-server-proxy env var set to empty'):
            with isolated_environ(empty=True, add=dict(
                DWAVE_INSPECTOR_JUPYTER_SERVER_PROXY_EXTERNAL_URL=''
            )):
                self.assertEqual(rewrite_url(local_url), local_url)

        with self.subTest('jupyter-server-proxy configured'):
            with isolated_environ(empty=True, add=dict(
                DWAVE_INSPECTOR_JUPYTER_SERVER_PROXY_EXTERNAL_URL=ext_base
            )):
                self.assertEqual(
                    rewrite_url(local_url),
                    f'{ext_base}proxy/{port}/?problemId=1'
                )
