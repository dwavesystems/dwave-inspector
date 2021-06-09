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

import sys
import time
import random
import logging
import traceback
import threading
from wsgiref.simple_server import make_server, WSGIRequestHandler, WSGIServer

try:
    # use a backport for python_version < 3.9
    import importlib_resources
except ImportError:
    import importlib.resources as importlib_resources

import requests
from flask import Flask, send_from_directory, make_response
from werkzeug.exceptions import NotFound

from dwave.inspector.storage import problem_store, problem_access_sem, get_problem, get_solver_data


# get local server/app logger
logger = logging.getLogger(__name__)

# suppress logging from Werkzeug
logging.getLogger('werkzeug').addHandler(logging.NullHandler(logging.DEBUG))


class LoggingStream(object):
    """Provide file-like interface to a logger."""

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        for line in message.split('\n'):
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        pass

# stream interface to our local logger
logging_stream = LoggingStream(logger, logging.DEBUG)


class LoggingWSGIRequestHandler(WSGIRequestHandler):
    """WSGIRequestHandler subclass that logs to our logger, instead of to
    ``sys.stderr`` (as hardcoded in ``http.server.BaseHTTPRequestHandler``).
    """

    def log_message(self, format, *args):
        logger.info(format, *args)

    def get_stderr(self):
        return logging_stream


class LoggingWSGIServer(WSGIServer):
    """WSGIServer subclass that logs to our logger, instead of to ``sys.stderr``
    (as hardcoded in ``socketserver.BaseServer.handle_error``).
    """

    def handle_error(self, request, client_address):
        traceback.print_exception(*sys.exc_info(), file=logging_stream)


class WSGIAsyncServer(threading.Thread):
    """WSGI server container for a wsgi app that runs asynchronously (in a
    separate thread).
    """

    def _safe_make_server(self, host, base_port, app, tries=20):
        """Instantiate a http server. Discover available port starting with
        `base_port` (use linear and random search).
        """

        def ports(start, linear=5):
            """Server port proposal generator. Starts with a linear search, then
            converts to a random look up.
            """
            for port in range(start, start + linear):
                yield port
            while True:
                yield random.randint(port + 1, (1<<16) - 1)

        for _, port in zip(range(tries), ports(start=base_port)):
            try:
                return make_server(host, port, app,
                                   server_class=LoggingWSGIServer,
                                   handler_class=LoggingWSGIRequestHandler)
            except OSError as exc:
                # handle only "[Errno 98] Address already in use"
                if exc.errno != 98:
                    raise

        raise RuntimeError("unable to find available port to bind local "
                           "webserver to even after {} tries".format(tries))

    def _make_server(self):
        # ensure inspector web app static data is available
        try:
            import dwave._inspectorapp as webappdata
        except ImportError:
            raise RuntimeError(
                "Cannot use the problem inspector without a non-open-source "
                "'inspector' application component. Try running "
                "'dwave install inspector' or consult the documentation.")

        self.app.webappdata = webappdata

        # create http server, and bind it to first available port >= base_port
        return self._safe_make_server(self.host, self.base_port, self.app)

    @property
    def server(self):
        """HTTP server accessor that creates the actual server instance
        (and binds it to host:port) on first access.
        """

        with self._server_lock:
            self._server = getattr(self, '_server', None)

            if self._server is None:
                self._server = self._make_server()

            return self._server

    def __init__(self, host, base_port, app):
        super(WSGIAsyncServer, self).__init__(daemon=True)

        # store config, but start the web server (and bind to address) on run()
        self.host = host
        self.base_port = base_port
        self.app = app
        self._server_lock = threading.RLock()

    def run(self):
        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()
        self.join()

    def get_inspect_url(self, problem_id):
        return 'http://{}:{}/?problemId={}'.format(
            *self.server.server_address, problem_id)

    def get_callback_url(self, problem_id):
        return 'http://{}:{}/api/callback/{}'.format(
            *self.server.server_address, problem_id)

    def _ensure_accessible(self, sleep=0.1, tries=100, timeout=10):
        """Ping the canary URL (app root) until the app becomes accessible."""

        canary = 'http://{}:{}/'.format(*self.server.server_address)

        for _ in range(tries):
            try:
                requests.get(canary, timeout=timeout).raise_for_status()
                return True
            except:
                time.sleep(sleep)

        return False

    def ensure_started(self):
        if not self.is_alive():
            self.start()
            return self._ensure_accessible()

        return True

    def ensure_stopped(self):
        if self.is_alive():
            self.stop()

    def wait_shutdown(self, timeout=None):
        logger.debug('%s.wait_shutdown(timeout=%r)', type(self).__name__, timeout)
        self.join(timeout)

    def wait_problem_accessed(self, problem_id, timeout=None):
        """Blocks until problem access semaphore is notified.

        Problem semaphore is created on access, so this method can be called
        even before the problem is created, or access is notified.
        """
        logger.debug('%s.wait_problem_accessed(problem_id=%r, timeout=%r)',
                     type(self).__name__, problem_id, timeout)
        problem_access_sem[problem_id].acquire(blocking=True, timeout=timeout)

    def notify_problem_accessed(self, problem_id):
        """Notifies problem access semaphore of one access (full load)."""
        logger.debug('%s.notify_problem_accessed(problem_id=%r)',
                     type(self).__name__, problem_id)
        problem_access_sem[problem_id].release()


app = Flask(__name__, static_folder=None)

@app.route('/')
@app.route('/<path:path>')
def send_static(path='index.html'):
    # NOTE: backport required for `.files` prior to py39
    basedir = importlib_resources.files(app.webappdata).joinpath('build')

    # NOTE: cast `basedir: PosixPath` to `str` to work on Flask@py35
    # NOTE: safe to do because inspectorapp (webappdata) is `zip_safe=False`!
    # XXX: remove when py35 is dropped; consider using werkzeug.safe_join directly
    return send_from_directory(str(basedir), path)

@app.route('/api/problems/<problem_id>')
def send_problem(problem_id):
    try:
        return problem_store[problem_id]
    except KeyError:
        raise NotFound

@app.route('/api/problems/<problem_id>/solver')
def send_solver(problem_id):
    try:
        return get_solver_data(problem_store[problem_id]['details']['solver'])
    except KeyError:
        raise NotFound

@app.route('/api/callback/<problem_id>')
def notify_problem_loaded(problem_id):
    # note: switch to POST to avoid caching issues altogether?
    app_server.notify_problem_accessed(problem_id)
    response = make_response(dict(ack=True))
    response.cache_control.no_store = True
    response.cache_control.max_age = 0  # force revalidation if already cached
    return response

@app.after_request
def add_header(response):
    # cache responses for a day, unless caching turned off
    if not response.cache_control.no_store:
        response.cache_control.public = True
        response.cache_control.max_age = 86400
    return response

app_server = WSGIAsyncServer(host='127.0.0.1', base_port=18000, app=app)
