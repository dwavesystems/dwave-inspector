.. image:: https://circleci.com/gh/dwavesystems/dwave-inspector.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dwave-inspector
    :alt: Linux/Mac build status


================
D-Wave Inspector
================

.. index-start-marker

A tool for visualizing problems submitted to (and answers received from) a
D-Wave structured solver.

.. index-end-marker


Installation or Building
========================

.. installation-start-marker

Install the closed-source dependencies::

    pip install dwave-inspectorapp --extra-index=https://pypi.dwavesys.com/simple

Then, install from package on D-Wave's PyPI::

    pip install dwave-inspector --extra-index=https://pypi.dwavesys.com/simple

or from source::

    pip install git+ssh://git@github.com/dwavesystems/dwave-inspector.git

Alternatively, clone and build from source::

    git clone https://github.com/dwavesystems/dwave-inspector.git
    cd dwave-inspector
    pip install -r requirements.txt
    python setup.py install

.. installation-end-marker


Example
=======

.. example-start-marker

The canonical way to use the Inspector is with samples in physical/qubit space.

.. code-block:: python

    import dwave.cloud
    import dwave.inspector

    # define problem
    h = {}
    J = {(0, 4): 1, (0, 5): 1, (4, 1): 1, (1, 5): -1}

    # get solver
    client = dwave.cloud.Client.from_config()
    solver = client.get_solver(qpu=True)

    # sample
    response = solver.sample_ising(h, J, num_reads=100)

    # inspect
    dwave.inspector.show((h, J), response)

It is possible to inspect QMIs given only samples in logical space, but the exact
response reconstruction is not possible in that case. Namely, chain breaks will
not be visible.

.. code-block:: python

    import dimod
    import dwave.inspector
    from dwave.system import DWaveSampler, EmbeddingComposite

    # define problem
    bqm = dimod.BQM.from_ising({}, {'ab': 1, 'bc': 1, 'ca': 1})

    # get sampler
    sampler = EmbeddingComposite(DWaveSampler(solver=dict(qpu=True)))

    # sample
    sampleset = sampler.sample(bqm)

    # inspect
    dwave.inspector.show(bqm, sampleset, sampler)

.. example-end-marker


License
=======

Released under the Apache License 2.0. See `<LICENSE>`_ file.
