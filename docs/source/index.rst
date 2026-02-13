🐟 Fishy Business
================

**Machine Learning for Rapid Evaporative Ionization Mass Spectrometry**

*A Doctoral Thesis by Jesse Wood*
*Victoria University of Wellington*

.. image:: https://readthedocs.org/projects/fishy-business/badge/?version=latest
   :target: https://fishy-business.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/woodRock/fishy-business/actions/workflows/black.yml/badge.svg
   :target: https://github.com/woodRock/fishy-business/actions/workflows/black.yml
   :alt: Format Python Code

.. image:: https://github.com/woodRock/fishy-business/actions/workflows/unittests.yml/badge.svg
   :target: https://github.com/woodRock/fishy-business/actions/workflows/unittests.yml
   :alt: Unit Tests

.. image:: https://coveralls.io/repos/github/woodRock/fishy-business/badge.svg?branch=main
   :target: https://coveralls.io/github/woodRock/fishy-business?branch=main
   :alt: Coverage Status

.. image:: https://github.com/woodRock/fishy-business/actions/workflows/doctests.yml/badge.svg
   :target: https://github.com/woodRock/fishy-business/actions/workflows/doctests.yml
   :alt: Doctests

A configuration-driven framework for analyzing mass spectrometry data using Deep Learning, Classic Machine Learning, and Evolutionary Algorithms.

.. important::
   While the source code of this framework is open-source under the MIT license, the accompanying **REIMS dataset is private**. Authorized users must use the provided download command to fetch the data files.

Quickstart
----------

Train a state-of-the-art Transformer model and view results in just 4 lines:

.. code-block:: python

   from fishy import TrainingConfig, run_unified_training, display_final_summary
   
   config = TrainingConfig(model="transformer", dataset="species")
   results = run_unified_training(config)
   display_final_summary(results)

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
   datasets
   wizard
   dashboard
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   concepts
   datasets_guide
   xai_guide
   papers
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api

.. toctree::
   :maxdepth: 1
   :caption: Community
   :hidden:

   contributing
   contact

Citation
--------

If you use this framework in your research, please cite the following paper:

.. code-block:: bibtex

   @article{wood2025hook,
     title={Hook, line, and spectra: machine learning for fish species identification and body part classification using rapid evaporative ionization mass spectrometry},
     author={Wood, Jesse and Nguyen, Bach and Xue, Bing and Zhang, Mengjie and Killeen, Daniel},
     journal={Intelligent Marine Technology and Systems},
     volume={3},
     number={1},
     pages={16},
     year={2025},
     publisher={Springer}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
