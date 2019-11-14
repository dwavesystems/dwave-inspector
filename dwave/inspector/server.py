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

from __future__ import absolute_import

import logging
import threading
from wsgiref.simple_server import make_server, WSGIRequestHandler

try:
    import importlib.resources as importlib_resources
except ImportError:
    # use a backport for python_version < 3.7
    import importlib_resources

from flask import Flask

try:
    import dwave._inspectorapp as appdata
except ImportError:
    # TODO: demote to warning only and use a dummy server in this case
    raise RuntimeError("Can't use the Inspector without 'dwave-inspectorapp' "
                       "package. Consult the docs for install instructions.")


# suppress logging from Flask app
logging.getLogger(__name__).addHandler(logging.NullHandler(logging.DEBUG))

# suppress logging from Werkzeug
logging.getLogger('werkzeug').addHandler(logging.NullHandler(logging.DEBUG))


class SilentWSGIRequestHandler(WSGIRequestHandler):
    """WSGIRequestHandler subclass with logging to ``sys.stderr`` suppressed.
    Alternatively, we can use 'logging' machinery instead.
    """

    # Note: this essentially fixes a design decision in the standard lib's
    # ``http.server.BaseHTTPRequestHandler`` to hardcode logging to stderr.

    def log_message(self, format, *args):
        pass


class WSGIAsyncServer(threading.Thread):
    """WSGI server container for a wsgi app that runs asynchronously (in a
    separate thread).
    """

    def __init__(self, host, port, app):
        super(WSGIAsyncServer, self).__init__(daemon=True)

        self.server = make_server(
            host, port, app, handler_class=SilentWSGIRequestHandler)

    def run(self):
        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()
        self.join()


app = Flask(__name__)

@app.route('/')
def _root():
    return 'root'

app_server = WSGIAsyncServer(host='127.0.0.1', port=8000, app=app)
