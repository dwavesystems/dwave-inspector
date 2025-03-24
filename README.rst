.. image:: https://badge.fury.io/py/dwave-inspector.svg
    :target: https://badge.fury.io/py/dwave-inspector
    :alt: Latest version on PyPI

.. image:: https://circleci.com/gh/dwavesystems/dwave-inspector.svg?style=shield
    :target: https://circleci.com/gh/dwavesystems/dwave-inspector
    :alt: Linux/MacOS build status

.. image:: https://codecov.io/gh/dwavesystems/dwave-inspector/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/dwavesystems/dwave-inspector
    :alt: Coverage report


===============
dwave-inspector
===============

.. start_inspector_about

A tool for visualizing problems submitted to, and answers received from, a
D-Wave structured solver such as an Advantage\ :sup:`TM` quantum computer.

This example shows a typical usage: a binary quadratic model minor-embedded onto
a quantum processing unit (QPU).

.. code-block:: python

    from dwave.system import DWaveSampler, EmbeddingComposite
    import dimod
    import dwave.inspector

    # Get sampler
    sampler = EmbeddingComposite(DWaveSampler())

    # Define a problem
    x, y, z = dimod.Binaries(['x', 'y', 'z'])
    bqm = x*y - x*z + 2*y

    # Sample
    sampleset = sampler.sample(bqm, num_reads=100)

    # Inspect
    dwave.inspector.show(sampleset)

.. end_inspector_about


Installation or Building
========================

If `Ocean SDK 2.0+ <https://github.com/dwavesystems/dwave-ocean-sdk>`_ is
installed:

.. code-block:: bash

    dwave install inspector

Otherwise, install the package from PyPI:

.. code-block:: bash

    pip install dwave-inspector

and then install the closed-source dependency with:

.. code-block:: bash

    pip install dwave-inspectorapp --extra-index=https://pypi.dwavesys.com/simple

Please note this closed-source dependency is released under the
`D-Wave EULA <https://docs.dwavequantum.com/en/latest/licenses.html>`_
license.

Alternatively, clone and build from source:

.. code-block:: bash

    git clone https://github.com/dwavesystems/dwave-inspector.git
    cd dwave-inspector
    pip install -r requirements.txt
    python setup.py install

When building from source, the closed-source component still needs to be
installed as above.


License
=======

Released under the Apache License 2.0. See LICENSE file.

Visualization component released under the
`D-Wave EULA <https://docs.dwavequantum.com/en/latest/licenses.html>`_.


Contributing
============

Ocean's `contributing guide <https://docs.dwavequantum.com/en/latest/ocean/contribute.html>`_
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
