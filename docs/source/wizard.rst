Interactive Wizard
==================

The easiest way to start using ``fishy-business`` is through the interactive setup wizard. 
It helps you select a model, choose a dataset, and configure your analysis flags without needing to remember CLI arguments.

To launch the wizard, run:

.. code-block:: bash

   python3 main.py wizard

How it works:
-------------

1. **Select Model Category**: Choose between Deep Learning, Classic ML, Evolutionary, etc.
2. **Select Dataset**: Pick one of the registered datasets (e.g., species, oil).
3. **Configure Flags**: Enable benchmarking, figure generation, or Weights & Biases logging.
4. **Output**: The wizard will provide you with the exact CLI command to run or save a YAML configuration file for you.
