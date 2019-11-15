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

Install closed-source dependencies:

    pip install dwave-inspectorapp --extra-index=https://pypi.dwavesys.com/simple

Then, install from package on PyPI::

    pip install dwave-inspector

or from source::

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
    sampler = EmbeddingComposite(DWaveSampler(solver='DW_2000Q_2_1'))

    # sample
    sampleset = sampler.sample(bqm, return_embedding=True)

    # inspect
    dwave.inspector.show(bqm, sampleset, sampler)

.. example-end-marker


License
=======

Released under the Apache License 2.0. See `<LICENSE>`_ file.
