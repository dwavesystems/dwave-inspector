.. image:: https://badge.fury.io/py/dwave-inspector.svg
    :target: https://badge.fury.io/py/dwave-inspector
    :alt: Latest version on PyPI

.. image:: https://circleci.com/gh/dwavesystems/dwave-inspector.svg?style=shield
    :target: https://circleci.com/gh/dwavesystems/dwave-inspector
    :alt: Linux/MacOS build status

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

    import dwave.system
    import dwave.inspector

    # Get sampler
    sampler = dwave.system.DWaveSampler()

    # Define a problem (actual qubits depend on the selected QPU's working graph)
    h = {}
    J = {(0, 4): 1, (0, 5): 1, (1, 4): 1, (1, 5): -1}
    assert all(edge in sampler.edgelist for edge in J)

    # Sample
    sampleset = sampler.sample_ising(h, J, num_reads=100)

    # Inspect
    dwave.inspector.show(sampleset)

.. example-end-marker


Installation or Building
========================

.. installation-start-marker

If `D-Wave Ocean SDK 2.0+ <https://docs.ocean.dwavesys.com/>`_ is installed:

.. code-block:: bash

    dwave install inspector

Otherwise, install the package from PyPI:

.. code-block:: bash

    pip install dwave-inspector

and then install the closed-source dependency with:

.. code-block:: bash

    pip install dwave-inspectorapp --extra-index=https://pypi.dwavesys.com/simple

Please note this closed-source dependency is released under the `D-Wave EULA`_ license.

Alternatively, clone and build from source:

.. code-block:: bash

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

Visualization component released under the `D-Wave EULA`_.

.. _D-Wave EULA: https://docs.ocean.dwavesys.com/projects/inspector/en/latest/license.html#inspector-eula


Contributing
============

Ocean's `contributing guide <https://docs.ocean.dwavesys.com/en/stable/contributing.html>`_
has guidelines for contributing to Ocean packages.

Release Notes
-------------

D-Wave Inspector uses `reno <https://docs.openstack.org/reno/>`_ to manage
its release notes.

When making a contribution to D-Wave Inspector that will affect users, create
a new release note file by running:

.. code-block:: bash

    reno new your-short-descriptor-here

You can then edit the file created under ``releasenotes/notes/``.
Remove any sections not relevant to your changes.
Commit the file along with your changes.

See reno's `user guide <https://docs.openstack.org/reno/latest/user/usage.html>`_
for details.
