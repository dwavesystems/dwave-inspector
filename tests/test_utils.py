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

import unittest

from uuid import uuid4
from urllib.parse import ParseResult, urlparse, urljoin

from dwave.inspector.utils import itemsgetter, annotated, update_url_from


class TestItemsgetter(unittest.TestCase):

    def test_nil(self):
        with self.assertRaises(TypeError):
            itemsgetter()

    def test_one(self):
        obj = list(range(3))
        f = itemsgetter(1)

        self.assertTrue(callable(f))
        self.assertEqual(f(obj), (obj[1], ))

    def test_multi(self):
        obj = list(range(3))
        f = itemsgetter(0, 2)

        self.assertTrue(callable(f))
        self.assertEqual(f(obj), (obj[0], obj[2]))


class TestAnnotation(unittest.TestCase):

    def test_nil(self):
        """Null annotation case correct."""

        f = lambda: None
        f_attrs = dir(f)

        f_ann = annotated()(f)
        f_ann_attrs = dir(f_ann)

        self.assertEqual(f_attrs, f_ann_attrs)

    def test_priority(self):
        """Single (priority) attribute set."""

        priority = 10

        @annotated(priority=priority)
        def f():
            pass

        self.assertEqual(f.priority, priority)

    def test_multiple(self):
        """Multiple function attributes are set."""

        attrs = dict(a=1, b=2, c=3)

        @annotated(**attrs)
        def f():
            pass

        for k, v in attrs.items():
            self.assertEqual(getattr(f, k), v)


class TestUrlRewrite(unittest.TestCase):

    def test_default(self):
        src, dst = 'http://localhost/', 'https://example.com/path'
        url = update_url_from(src, dst)
        self.assertEqual(url, dst)

    def test_partial(self):
        for field in ParseResult._fields:
            with self.subTest(field):
                val = str(uuid4())
                val = f'/{val}' if field == 'path' else val
                val = f'a{val}' if field == 'scheme' else val
                res = urlparse(update_url_from('', '', **{field: lambda *args: val}))
                self.assertEqual(getattr(res, field), val)

    def test_mix(self):
        src, dst = 'http://localhost:9000/', 'https://example.com/path'
        url = update_url_from(src, dst, netloc=lambda src, dst: f'{src.port}-{dst.hostname}')
        self.assertEqual(url, 'https://9000-example.com/path')

    def test_path_join(self):
        src, dst = 'http://localhost:9000/local', 'https://example.com/external/'
        url = update_url_from(src, dst, path=lambda src, dst: urljoin(dst.path, src.path.lstrip('/')))
        self.assertEqual(url, 'https://example.com/external/local')
