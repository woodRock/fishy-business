Fishy Business: Spectral Analysis Framework
===========================================

.. image:: https://coveralls.io/repos/github/woodRock/fishy-business/badge.svg?branch=main
   :target: https://coveralls.io/github/woodRock/fishy-business?branch=main
   :alt: Coverage Status

A configuration-driven framework for analyzing mass spectrometry data using Deep Learning, Classic Machine Learning, and Evolutionary Algorithms.

3-Line Quickstart
-----------------

Train a state-of-the-art Transformer model on your spectral data in just a few lines:

.. code-block:: python

   from fishy import TrainingConfig, run_unified_training
   
   config = TrainingConfig(model="transformer", dataset="species", file_path="data/REIMS.xlsx")
   results = run_unified_training(config)

Key Features
------------

* **Universal API**: Use the same interface for PyTorch, Scikit-Learn, and DEAP models.
* **Auto-Validation**: Built-in K-Fold cross-validation and statistical significance testing.
* **Research Ready**: Specialized support for pre-training, transfer learning, and contrastive suites.
* **XAI Integrated**: Visual explanations using Grad-CAM and LIME out of the box.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   wizard
   examples

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   papers

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api

.. toctree::
   :maxdepth: 1
   :caption: Community
   :hidden:

   contact

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
