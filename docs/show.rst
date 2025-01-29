.. _show_inspector:

====================
Visualizing Problems
====================

Typically you use the :func:`~dwave.inspector.show` function on a
:class:`~dimod.SampleSet` returned from the quantum computer or on the SAPI
problem ID\ [#active-problem]_. Other problem inputs, such as the binary
quadratic model---in BQM, Ising, or QUBO formats---and an embedding, are
optional.
However, to visualize a logical problem if
:std:doc:`dimod <oceandocs:docs_dimod/sdk_index>`\ 's
:class:`~dwave.system.composites.EmbeddingComposite` or derived classes are not
used, you must supply the embedding.

Below are some options for providing problem data to the
:func:`~dwave.inspector.show` function, where
``response`` was returned for a problem defined directly on physical qubits and
``sampleset`` returned from a problem submitted using
:class:`~dwave.system.composites.EmbeddingComposite`:

.. code-block:: python

    show(response)
    show('69ace80c-d3b1-448a-a028-b51b94f4a49d')   # Using a SAPI problem ID
    show((h, J), response)
    show(Q, response)
    show((h, J), response, dict(embedding=embedding, chain_strength=5))

    show(sampleset)
    show(bqm, sampleset)

To see detailed parameter information, see the relevant function below.

The :func:`~dwave.inspector.show` function supports flow control for scripts
with the ``block`` parameter. For example, the default setting of ``once``
(``dwave.inspector.Block.ONCE``) blocks until your problem is loaded from the
inspector web server and ``forever`` blocks until you terminate with a
Ctrl+C/SIGTERM.

.. rubric:: Footnotes

.. [#active-problem]
    For problems submitted in the active session (i.e., once the problem
    inspector has been imported).

.. currentmodule:: dwave.inspector

Classes
=======

.. autoclass:: Block

Functions
=========

.. autosummary::
    :toctree: generated/

    show
    show_bqm_response
    show_bqm_sampleset
    show_qmi
