Contributing
============

We welcome contributions! Whether you're adding a new deep learning architecture or a baseline model, here's how to get started.

Adding a New Model
------------------

1.  **Implement the Module**:
    Place your model in the appropriate directory:
    *   Deep Learning: ``fishy/models/deep/``
    *   Classic ML: ``fishy/models/classic/``
    *   Probabilistic: ``fishy/models/probabilistic/``

2.  **Register the Model**:
    Open ``fishy/configs/models.yaml`` and add your model class path and default hyperparameters.

3.  **Update the Factory**:
    If your model requires a unique initialization signature (e.g., specific number of heads), add a condition to ``fishy/_core/factory.py``.

Development Workflow
--------------------

*   **Linting**: We use ``black`` for formatting and ``pylint`` for code quality.
*   **Testing**: Ensure any new code has unit tests in the ``tests/`` folder and passing doctests in the docstrings.
*   **Documentation**: If you add a new high-level feature, consider adding a tutorial in the ``examples/`` folder.

Running Tests
-------------

.. code-block:: bash

   # Run all unit tests
   pytest tests/

   # Run all docstring tests
   pytest --doctest-modules fishy/
