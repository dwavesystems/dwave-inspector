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
from unittest import mock

from dwave.inspector.viewers import (
    prioritized_viewers, webbrowser_tab, webbrowser_window, view)


class DummyZMQInteractiveShell:
    pass


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

    @mock.patch('dwave.inspector.viewers.get_ipython', DummyZMQInteractiveShell, create=True)
    @mock.patch('dwave.inspector.viewers.display', lambda o: None, create=True)
    def test_hijack(self):
        """jupyter_inline viewer prevents blocking."""

        self.assertEqual(view('url'), False)

    @mock.patch('dwave.inspector.viewers.get_ipython', DummyZMQInteractiveShell, create=True)
    @mock.patch('dwave.inspector.viewers.display', create=True)
    def test_hijack(self, display):
        """jupyter_inline viewer calls display."""

        url = 'url'
        self.assertEqual(view(url), False)
        display.assert_called_once_with(url)

    @mock.patch('webbrowser.open_new_tab', return_value='webbrowser_tab')
    @mock.patch('dwave.inspector.viewers.get_ipython', object, create=True)
    def test_terminal_ipython(self, m):
        """jupyter_inline viewer ignores non-gui ipython."""

        self.assertEqual(view('url'), 'webbrowser_tab')

    @mock.patch('webbrowser.open_new_tab', side_effect=ValueError)
    @mock.patch('webbrowser.open_new', side_effect=ValueError)
    @mock.patch('dwave.inspector.viewers.get_ipython', side_effect=ValueError, create=True)
    def test_nonblocking_view(self, m1, m2, m3):
        """Signal non-blocking show behavior when no viewer succeeds."""

        self.assertEqual(view('url'), False)
