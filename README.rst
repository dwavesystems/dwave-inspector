.. image:: https://badge.fury.io/py/dwave-inspector.svg
    :target: https://badge.fury.io/py/dwave-inspector
    :alt: Last version on PyPI

.. image:: https://circleci.com/gh/dwavesystems/dwave-inspector.svg?style=shield
    :target: https://circleci.com/gh/dwavesystems/dwave-inspector
    :alt: Linux/Mac build status

.. image:: https://codecov.io/gh/dwavesystems/dwave-inspector/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/dwavesystems/dwave-inspector
    :alt: Coverage report

.. image:: https://readthedocs.com/projects/d-wave-systems-dwave-inspector/badge/?version=latest
    :target: https://docs.ocean.dwavesys.com/projects/inspector/en/latest/?badge=latest
    :alt: Documentation Status

================
D-Wave Inspector
================

.. index-start-marker

A tool for visualizing problems submitted to, and answers received from, a
D-Wave structured solver such as a D-Wave 2000Q quantum computer.

.. index-end-marker

Example
=======

.. example-start-marker

This example shows the canonical usage: samples representing physical qubits on
a quantum processing unit (QPU).

.. code-block:: python

    from dwave.system import DWaveSampler
    import dwave.inspector

    # Get solver
    sampler = DWaveSampler(solver=dict(qpu=True))

    # Define a problem (actual qubits depend on the selected QPU's working graph)
    h = {}
    J = {(0, 4): 1, (0, 5): 1, (1, 4): 1, (1, 5): -1}
    assert all(edge in sampler.edgelist for edge in J)

    # Sample
    response = sampler.sample_ising(h, J, num_reads=100)

    # Inspect
    dwave.inspector.show(response)

.. example-end-marker

Installation or Building
========================

.. installation-start-marker

If `D-Wave Ocean SDK 2.0+ <https://docs.ocean.dwavesys.com/>`_ is installed::

    dwave install inspector

Otherwise, install the package from PyPI::

    pip install dwave-inspector

and then install the closed-source dependencies with::

    pip install dwave-inspectorapp --extra-index=https://pypi.dwavesys.com/simple

Alternatively, clone and build from source::

    git clone https://github.com/dwavesystems/dwave-inspector.git
    cd dwave-inspector
    pip install -r requirements.txt
    python setup.py install

When building from source, the closed-source component still needs to be
installed as above.

.. installation-end-marker

License
=======

Released under the Apache License 2.0. See `<LICENSE>`_ file.
