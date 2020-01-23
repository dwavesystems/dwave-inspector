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


Overview
========

``dwave-inspector`` provides a graphic interface for examining D-Wave quantum computers'
problems and answers. As described in the
`Ocean documentation's Getting Started <https://docs.ocean.dwavesys.com/en/latest/overview/solving_problems.html>`_,
the D-Wave system solves problems formulated as binary quadratic models (BQM) that are
mapped to its qubits in a process called minor-embedding. Because the way you choose to
minor-embed a problem (the mapping and related parameters) affects solution quality,
it can be helpful to see it.

For example, embedding a :math:`K_3` fully-connected graph, such as the
`Boolean AND gate example <https://docs.ocean.dwavesys.com/en/latest/examples/and.html>`_,
into a D-Wave 2000Q, with its Chimera topology, requires representing one of the
three variables with a "chain" of two physical qubits:

.. figure:: _images/and_gate.png
  :align: center
  :figclass: align-center
  :scale: 35%

  The AND gate's original BQM is represented on the left; its embedded representation, on the right, shows a two-qubit chain of qubits 1195 and 1199 for variable X1.

The problem inspector shows you your chains at a glance: you see lengths, any breakages,
and physical layout.



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

It is also possible to inspect QMIs given only samples in logical space:

.. code-block:: python

    import dimod
    import dwave.inspector
    from dwave.system import DWaveSampler, EmbeddingComposite

    # define problem
    bqm = dimod.BQM.from_ising({}, {'ab': 1, 'bc': 1, 'ca': 1})

    # get sampler
    sampler = EmbeddingComposite(DWaveSampler(solver=dict(qpu=True)))

    # sample
    sampleset = sampler.sample(bqm, num_reads=100)

    # inspect
    dwave.inspector.show(bqm, sampleset)

.. example-end-marker


License
=======

Released under the Apache License 2.0. See `<LICENSE>`_ file.
