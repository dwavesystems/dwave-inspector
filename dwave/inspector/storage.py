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

import threading
from collections import OrderedDict, defaultdict


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


class ProblemData(object):
    # QMI/problem submitted, dict with keys: linear, quadratic, type_, params
    problem = None

    # dwave.cloud.solver.StructuredSolver instance
    solver = None

    # dwave.cloud.computation.Future instance
    response = None

    def __init__(self, problem, solver, response):
        self.problem = problem
        self.solver = solver
        self.response = response


def add_problem(problem, solver, response):
    # store the problem encapsulated with ProblemData
    pd = ProblemData(problem=problem, solver=solver, response=response)
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

    # add them to the indexed collection
    # and remove them from the input bag
    for pd in resolved:
        problemdata[pd.response.id] = pd
        problemdata_bag.remove(pd)


def get_problem(problem_id):
    """Return :class:`.ProblemData` from problem data store, or fail with
    :exc:`KeyError`.
    """
    if problem_id not in problemdata:
        index_resolved_problems()

    return problemdata[problem_id]


def get_solver_data(solver_id):
    """Return solver data dict for `solver_id`. If solver hasn't been seen in
    any of the problems cached so far, fail with :exc:`KeyError`.
    """
    if solver_id in solvers:
        return solvers[solver_id].data

    raise KeyError('solver not found')
