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

import importlib.resources
import logging
import time
from typing import Optional
from urllib.parse import urljoin


import requests
from flask import Flask, send_from_directory, make_response, request, redirect
from werkzeug.exceptions import NotFound

from dwave.cloud.auth.server import BackgroundAppServer

from dwave.inspector.config import config
from dwave.inspector.storage import (
    problem_store, problem_access_sem,
    get_last_problem_id, get_solver_data)
from dwave.inspector.utils import OrJSONProvider


# get local server/app logger
logger = logging.getLogger(__name__)

# suppress logging from Werkzeug
logging.getLogger('werkzeug').addHandler(logging.NullHandler(logging.DEBUG))


class InspectorAppServer(BackgroundAppServer):
    """An extension of :class:`~dwave.cloud.auth.server.BackgroundAppServer`
    that provides inspector-specific control over background server termination.
    """

    def _make_server(self):
        # ensure inspector web app static data is available before we
        # create the server (that starts the app on request)
        try:
            import dwave._inspectorapp as webappdata
        except ImportError:
            raise RuntimeError(
                "Cannot use the problem inspector without a non-open-source "
                "'inspector' application component. Try running "
                "'dwave install inspector' or consult the documentation.")

        self.app.webappdata = webappdata

        # proceed with server creation
        return super()._make_server()

    def ensure_started(self, timeout: Optional[float] = None):
        if not self.is_alive():
            self.start()
            self.wait_ready(timeout=timeout)
            return self.wait_app_alive()

        return True

    def wait_app_alive(self, sleep: float = 0.1, tries: int = 100, timeout: float = 10):
        """Ping the canary URL (`/ping`) until the app becomes responsive."""

        canary = urljoin(self.root_url, '/ping')
        for _ in range(tries):
            try:
                requests.get(canary, timeout=timeout).raise_for_status()
                return True
            except:
                time.sleep(sleep)

        return False

    def wait_problem_accessed(self, problem_id: str, timeout: Optional[float] = None):
        """Blocks until problem access semaphore is notified, or timeout exceeded,
        in which case it raises a :exc:`TimeoutError`.

        Problem semaphore is created on access, so this method can be called
        even before the problem is created, or access is notified.
        """
        logger.debug('%s.wait_problem_accessed(problem_id=%r, timeout=%r)',
                     type(self).__name__, problem_id, timeout)

        acquired = problem_access_sem[problem_id].acquire(blocking=True, timeout=timeout)

        if timeout is not None and not acquired:
            raise TimeoutError("Problem not accessed within the specified timeout.")

    def notify_problem_accessed(self, problem_id: str):
        """Notifies problem access semaphore of one access (full load).
        """
        logger.debug('%s.notify_problem_accessed(problem_id=%r)',
                     type(self).__name__, problem_id)

        problem_access_sem[problem_id].release()

    def get_inspect_url(self, problem_id):
        return urljoin(self.root_url, f'/?problemId={problem_id}')

    def get_callback_url(self, problem_id):
        return urljoin(self.root_url, f'/api/callback/{problem_id}')

    def get_problem_url(self, problem_id):
        return urljoin(self.root_url, f'/api/problems/{problem_id}')

    def get_solver_url(self, problem_id):
        return urljoin(self.root_url, f'/api/problems/{problem_id}/solver')


app = Flask(__name__, static_folder=None)
app.json = OrJSONProvider(app)

@app.route('/ping')
def ping():
    return 'pong'

@app.route('/')
@app.route('/<path:path>')
def send_static(path=None):
    # handle with missing problem id by redirecting to the last problem stored
    problem_id = request.args.get('problemId')
    if path is None and problem_id is None:
        last_problem_id = get_last_problem_id()
        if last_problem_id is not None:
            logger.debug('redirecting to last_problem_id=%r', last_problem_id)
            response = redirect(f'/?problemId={last_problem_id}')
            # make sure we don't cache this (see `add_header()` below)
            response.cache_control.no_store = True
            return response

    basedir = importlib.resources.files(app.webappdata).joinpath('build')

    # NOTE: safe to do because inspectorapp (webappdata) is `zip_safe=False`!
    path = 'index.html' if path is None else path
    return send_from_directory(basedir, path)

@app.route('/api/problems/<problem_id>')
def send_problem(problem_id):
    try:
        return problem_store[problem_id]
    except KeyError:
        raise NotFound

@app.route('/api/problems/<problem_id>/solver')
def send_solver(problem_id):
    try:
        return get_solver_data(problem_store[problem_id]['details']['solver'], update_inplace=True)
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

app_server = InspectorAppServer(
    host=config.host,
    base_port=config.base_port,
    max_port=config.max_port,
    linear_tries=config.port_search_linear_tries,
    randomized_tries=config.port_search_randomized_tries,
    app=app
)
