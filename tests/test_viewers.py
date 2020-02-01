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

import mock
import unittest

from dwave.inspector.viewers import (
    annotated, prioritized_viewers, webbrowser_tab, webbrowser_window, view)


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


class TestViewers(unittest.TestCase):

    def test_registration(self):
        """Default viewers are registered with correct priorities."""

        local_viewers = {webbrowser_tab, webbrowser_window}

        all_viewers = set(prioritized_viewers())

        for lv in local_viewers:
            self.assertIn(lv, all_viewers)

    def test_prioritization(self):
        """Viewers correctly ordered (by desc priority)."""

        pri = [v.priority for v in prioritized_viewers()]

        self.assertEqual(pri, sorted(pri, reverse=True))

    @mock.patch('webbrowser.open_new_tab', return_value='webbrowser_tab')
    def test_preferred(self, m):
        """Preferred viewer is called."""

        self.assertEqual(view('url'), 'webbrowser_tab')

    @mock.patch('webbrowser.open_new_tab', side_effect=ValueError)
    @mock.patch('webbrowser.open_new', return_value='webbrowser_window')
    def test_fallback(self, m1, m2):
        """Fallback to secondary viewer works when primary fails."""

        self.assertEqual(view('url'), 'webbrowser_window')
