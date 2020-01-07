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

Then, install from package on PyPI::

    pip install dwave-inspector

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

.. code-block:: python

    import dimod
    import dwave.inspector
    from dwave.system import DWaveSampler, EmbeddingComposite

    # define problem
    bqm = dimod.BQM.from_ising({}, {'ab': 1, 'bc': 1, 'ca': 1})

    # get sampler
    sampler = EmbeddingComposite(DWaveSampler(solver=dict(qpu=True)))

    # sample
    sampleset = sampler.sample(bqm, return_embedding=True)

    # inspect
    dwave.inspector.show(bqm, sampleset, sampler)

.. example-end-marker


Known Issues
============

- Only one instance of the Inspector can be active at a time. If running examples, exit each before running the next one.
- Warnings not available yet. Pending changes in `dwave-system`.
- Debug/error output of the background HTTP server not always suppressed.
- Only Leap solvers can be used (2000Q).

License
=======

Released under the Apache License 2.0. See `<LICENSE>`_ file.
