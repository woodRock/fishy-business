XAI & Interpretability
======================

Mass spectrometry models are often "black boxes." This framework integrates Explainable AI (XAI) tools to help you understand which peaks drive a model's prediction.

Grad-CAM (1D)
-------------
Grad-CAM (Gradient-weighted Class Activation Mapping) is typically used for images, but we adapt it here for 1D spectra. It calculates the importance of each feature by looking at the gradients of the target class flowing into the last convolutional or attention layer.

**Interpretation**:
*   A high "activation" score at a specific m/z value indicates that the model heavily relied on that peak to make its prediction.
*   In the generated plots, peaks highlighted in red/yellow are "driving" the classification.

LIME
----
LIME (Local Interpretable Model-agnostic Explanations) creates a local linear approximation of the model around a specific sample.

**When to use**:
*   Use Grad-CAM for deep learning models (CNNs, Transformers).
*   Use LIME for any model (including OPLS-DA or Random Forest) to see a human-readable list of top-weighted features for a single prediction.

Generating Explanations
-----------------------
You can trigger XAI analysis by adding the ``--xai`` flag to the training command:

.. code-block:: bash

   fishy train -m transformer -d species --xai

Plots will be saved in the ``figures/`` subdirectory of your experiment run.
