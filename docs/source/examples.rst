Examples and Tutorials
=======================

This section contains a step-by-step tutorial on how to use the `fishy-business` library programmatically.

Getting Started
---------------
The simplest way to run a training experiment using the high-level `run_unified_training` interface.

.. literalinclude:: ../../examples/01_getting_started.py
   :language: python
   :linenos:

DataModule and Processing
-------------------------
Learn how the `DataModule` handles data loading, filtering, and conversion into PyTorch-ready tensors.

.. literalinclude:: ../../examples/02_data_module.py
   :language: python
   :linenos:

Configuration Management
------------------------
Using `TrainingConfig` and `ExperimentConfig` to centralize hyperparameters and experimental settings.

.. literalinclude:: ../../examples/03_configuration.py
   :language: python
   :linenos:

Training Engines
----------------
Exploring different ways to train models, from automated orchestration to direct control over the training loop.

.. literalinclude:: ../../examples/04_training_engines.py
   :language: python
   :linenos:

Automated Benchmarking
----------------------
How to trigger the full automated benchmark suite used in research papers.

.. literalinclude:: ../../examples/05_run_all_benchmarks.py
   :language: python
   :linenos:

Self-Supervised Pre-training
----------------------------
Demonstrates how to use unlabeled or semi-labeled data to pre-train a model using various self-supervised tasks.

.. literalinclude:: ../../examples/06_pretraining.py
   :language: python
   :linenos:

Sequential Transfer Learning
----------------------------
How to transfer knowledge from one dataset to another sequentially, using different classes/tasks at each stage.

.. literalinclude:: ../../examples/07_transfer_learning.py
   :language: python
   :linenos:

Probabilistic Inference
-----------------------
Using Bayesian models like Gaussian Processes to get predictions along with uncertainty estimates.

.. literalinclude:: ../../examples/08_probabilistic_inference.py
   :language: python
   :linenos:

Outputs and Visualization
-------------------------
Where to find experiment results and how to interpret generated artifacts like logs, metrics, and figures.

.. literalinclude:: ../../examples/09_outputs_and_visualization.py
   :language: python
   :linenos:
