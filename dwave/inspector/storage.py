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

import time
import logging
import threading
import functools
from collections import OrderedDict, defaultdict

from dwave.cloud.solver import StructuredSolver

from dwave.inspector.adapters import solver_data_postprocessed

logger = logging.getLogger(__name__)


# dict[problem_id: str, problem_data: dict]
problem_store = OrderedDict()

# dict[problem_id: str, accessed: threading.Semaphore]
problem_access_sem = defaultdict(lambda: threading.Semaphore(value=0))


def push_inspector_data(data):
    # push solver
    add_solver(data['rel']['solver'])
    del data['rel']

    # push problem data
    id_ = data['details']['id']
    problem_store[id_] = data

    return id_


# captures QMI + computation (response future)
problemdata_bag = set()
problemdata = {}

# map of all solvers seen in problems processed so far
solvers = {}


class ProblemData:
    # QMI/problem submitted, dict with keys: linear, quadratic, type_, params
    problem = None

    # dwave.cloud.solver.StructuredSolver instance
    solver = None

    # dwave.cloud.computation.Future instance
    response = None

    def __init__(self, problem, solver, response):
        if not isinstance(solver, StructuredSolver):
            raise TypeError('structured solver required')
        self.solver = solver

        if 'linear' not in problem or 'quadratic' not in problem:
            raise TypeError('invalid problem structure')
        self.problem = problem
        self.response = response

    def __eq__(self, other):
        return (self.problem is other.problem and
                self.solver is other.solver and
                self.response is other.response)


@functools.total_ordering
class ProblemDataTimestamped(ProblemData):
    """ProblemData tagged with creation timestamp and ordered chronologically."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.created_at = time.monotonic()

    def __lt__(self, other):
        return self.created_at < other.created_at

    def __eq__(self, other):
        return self.created_at == other.created_at and super().__eq__(other)

    def __hash__(self):
        return id(self)


def add_problem(problem, solver, response):
    logger.debug('storage.add_problem(problem=%r, solver=%r, response=%r)',
                 problem, solver, response)

    # store the problem encapsulated with ProblemData
    pd = ProblemDataTimestamped(problem=problem, solver=solver, response=response)
    problemdata_bag.add(pd)

    # cache solver reference
    add_solver(solver)


def add_solver(solver):
    solvers[solver.id] = solver


def index_resolved_problems():
    """Move problems that have `problem_id` assigned from `problemdata_bag` to
    `problemdata` dict.
    """

    # find problems that have id assigned
    resolved = set()
    for pd in problemdata_bag:
        if pd.response.id is not None:
            resolved.add(pd)

    # add them to the indexed collection (favorize newer problems)
    # and remove them from the input bag
    for pd in resolved:
        problem_id = pd.response.id
        if problem_id in problemdata:
            logger.debug('duplicate problem data (by problem_id) found; '
                         'discarding the older one')
            problemdata[problem_id] = max(pd, problemdata[problem_id])
        else:
            problemdata[problem_id] = pd
        problemdata_bag.remove(pd)


def get_problem(problem_id):
    """Return :class:`.ProblemData` from problem data store, or fail with
    :exc:`KeyError`.
    """
    # always re-index so newer problems with duplicate ids are preferred
    index_resolved_problems()

    # possibly fail with KeyError
    return problemdata[problem_id]


def get_solver_data(solver_id):
    """Return solver data dict for `solver_id`. If solver hasn't been seen in
    any of the problems cached so far, fail with :exc:`KeyError`.
    """
    if solver_id in solvers:
        return solver_data_postprocessed(solvers[solver_id])

    raise KeyError('solver not found')
