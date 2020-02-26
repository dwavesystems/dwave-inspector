.. image:: https://badge.fury.io/py/dwave-inspector.svg
    :target: https://badge.fury.io/py/dwave-inspector
    :alt: Last version on PyPI

.. image:: https://circleci.com/gh/dwavesystems/dwave-inspector.svg?style=shield
    :target: https://circleci.com/gh/dwavesystems/dwave-inspector
    :alt: Linux/Mac build status

.. image:: https://codecov.io/gh/dwavesystems/dwave-inspector/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/dwavesystems/dwave-inspector
    :alt: Coverage report


================
D-Wave Inspector
================

.. index-start-marker

A tool for visualizing problems submitted to, and answers received from, a
D-Wave structured solver such as a D-Wave 2000Q quantum computer.

.. index-end-marker

* `Overview`_
* `Installation or Building`_
* `Usage and Examples`_

.. _overview_inspector:

Overview
========

.. overview-start-marker

``dwave-inspector`` provides a graphic interface for examining D-Wave quantum computers'
problems and answers. As described in the
`Ocean documentation's Getting Started <https://docs.ocean.dwavesys.com/en/latest/overview/solving_problems.html>`_,
the D-Wave system solves problems formulated as binary quadratic models (BQM) that are
mapped to its qubits in a process called minor-embedding. Because the way you choose to
minor-embed a problem (the mapping and related parameters) affects solution quality,
it can be helpful to see it.

For example, embedding a K3 fully-connected graph, such as the
`Boolean AND gate example <https://docs.ocean.dwavesys.com/en/latest/examples/and.html>`_
into a D-Wave 2000Q, with its Chimera topology,
requires representing one of the three variables with a "chain" of two physical qubits:

.. figure:: https://raw.githubusercontent.com/dwavesystems/dwave-inspector/master/docs/_images/and_gate.png
  :align: center
  :figclass: align-center

  The AND gate's original BQM is represented on the left; its embedded representation,
  on the right, shows a two-qubit chain of qubits 1195 and 1199 for one variable.

The problem inspector shows you your chains at a glance: you see lengths, any breakages,
and physical layout.

.. overview-end-marker

.. _install_inspector:

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

.. _examples_inspector:

Usage and Examples
==================

.. usage-start-marker

Import the problem inspector to enable it\ [#]_ to hook into your problem submissions.

.. [#]
   Importing the problem inspector activates for the session the capture of
   data such as problems sent to the QPU and returned responses, relevant details of
   minor-embedding, and warnings. The recommended workflow is to import it at the
   start of your coding session as is typical for Python packages (it is also
   possible, but less convenient, to specify in the submission that
   data such as embedding be returned with the response).

Use the ``show()`` method to visualize the embedded problem, and optionally the
logical problem, in your default browser.

* `Inspecting an Embedded Problem`_
* `Inspecting a Logical Problem`_
* `show() Method`_

Inspecting an Embedded Problem
------------------------------

This example shows the canonical usage: samples representing physical qubits on a
quantum processing unit (QPU).

>>> from dwave.system import DWaveSampler
>>> import dwave.inspector
...
>>> # Get solver
>>> sampler = DWaveSampler(solver = {'qpu': True})
...
>>> # Define a problem (actual qubits depend on the selected QPU's working graph)
>>> h = {}
>>> all (edge in sampler.edgelist for edge in {(0, 4), (0, 5), (1, 4), (1, 5)})
True
>>> J = {(0, 4): 1, (0, 5): 1, (1, 4): 1, (1, 5): -1}
...
>>> # Sample
>>> response = sampler.sample_ising(h, J, num_reads=100)
...
>>> # Inspect
>>> dwave.inspector.show(response)

.. figure:: https://raw.githubusercontent.com/dwavesystems/dwave-inspector/master/docs/_images/physical_qubits.png
  :align: center
  :figclass: align-center

  Edge values between qubits 0, 1, 4, 5, and the selected solution, are shown by color on the left; a histogram, on the right, shows the energies of returned samples.

Inspecting a Logical Problem
----------------------------

This example visualizes a problem specified logically and then automatically
minor-embedded by Ocean's ``EmbeddingComposite``. For illustrative purposes
it sets a weak ``chain_strength`` to show broken chains.

.. code-block:: python

    import dimod
    import dwave.inspector
    from dwave.system import DWaveSampler, EmbeddingComposite

    # Define problem
    bqm = dimod.BQM.from_ising({}, {'ab': 1, 'bc': 1, 'ca': 1})

    # Get sampler
    sampler = EmbeddingComposite(DWaveSampler(solver=dict(qpu=True)))

    # Sample with low chain strength
    sampleset = sampler.sample(bqm, num_reads=1000, chain_strength=0.1)

    # Inspect
    dwave.inspector.show(sampleset)

.. figure:: https://raw.githubusercontent.com/dwavesystems/dwave-inspector/master/docs/_images/logical_problem.png
  :align: center
  :figclass: align-center

  The logical problem, on the left, shows that the value for variable ``b`` is based on a broken chain; the embedded problem, on the right, highlights the broken chain (its two qubits have different values) in bold red.

``show()`` Method
-----------------

The ``show()`` method requires the ``SampleSet`` returned from the quantum computer
or the SAPI problem ID\ [#]_\ ; other problem inputs---the binary quadratic model in BQM, Ising,
or QUBO formats, and an emebedding---are optional. However, to visualize a logical problem
if *dimod's* ``EmbeddingComposite`` or derived classes are not used, you must supply
the embedding.

.. [#]
   For problems submitted in the active session (i.e., once the problem inspector has been imported).

Below are some options for providing problem data to the ``show()`` method, where
``response`` was returned for a problem defined directly on physical qubits and
``sampleset`` returned from a problem submitted using ``EmbeddingComposite``:

.. code-block:: python

    show(response)
    show('69ace80c-d3b1-448a-a028-b51b94f4a49d')   # Using a SAPI problem ID
    show((h, J), response)
    show(Q, response)
    show((h, J), response, dict(embedding=embedding, chain_strength=5))

    show(sampleset)
    show(bqm, sampleset)

The ``show()`` method supports flow control for scripts with the ``block`` parameter.
For example, the default setting of ``once`` (``dwave.inspector.Block.ONCE``) blocks
until your problem is loaded from the inspector web server and ``forever`` blocks
until you terminate with a CNTL-C/SIGTERM.

.. usage-end-marker

License
=======

Released under the Apache License 2.0. See `<LICENSE>`_ file.
