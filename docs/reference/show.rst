.. _show_inspector:

===========================
Simulated Annealing Sampler
===========================


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
