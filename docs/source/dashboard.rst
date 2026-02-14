Interactive Dashboard
=====================

The framework includes a comprehensive web-based dashboard built with Streamlit for interactive data exploration, training monitoring, and biomarker discovery.

Launching the Dashboard
-----------------------

To start the dashboard, run the following command from the root of the repository:

.. code-block:: bash

   fishy dashboard

Once running, the dashboard will be available in your browser (typically at ``http://localhost:8501``).

Key Features
------------

Data Exploration
~~~~~~~~~~~~~~~~
* **Interactive Spectrum Viewer**: Filter and overlay individual samples to compare spectral signatures across classes.
* **Cluster Analysis**: Visualize data topology using linear (PCA) and non-linear (t-SNE, UMAP) dimensionality reduction.
* **Statistical Insights**: View mean spectral signatures with standard deviation shading and class-wise intensity distributions.

Training & Results
~~~~~~~~~~~~~~~~~~
* **Real-time Monitoring**: Watch training progress with live loss and accuracy curves.
* **Performance Analysis**: Detailed confusion matrices, ROC/AUC curves, and class-specific precision-recall metrics.
* **Error Analysis**: Use the "Misclassification Spotlight" to identify and visualize samples where the model was confidently wrong.

Interpretability & Biomarkers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Single Instance XAI**: Explain individual predictions using Grad-CAM (for deep models) or LIME.
* **Biomarker Correlation Network**: Visualize statistical links between top diagnostic peaks (r > 0.8) to identify redundant chemical drivers (isotopes/adducts).
* **Biomarker Stability Histogram**: Track the frequency of specific m/z features as top-20 diagnostic markers across multiple samples.
* **Class Comparison**: Highlight unique biomarkers in Gold/Silver directly on class-representative spectra to verify their alignment with physical peaks.

Leaderboard & Remote Data
~~~~~~~~~~~~~~~~~~~~~~~~~
* **SSH Proxy Jump**: Seamlessly aggregate results from remote high-performance servers (e.g., VUW ECS) through 2FA-secured jump hosts.
* **Data Persistence**: Save snapshots of the global leaderboard locally for offline analysis and thesis reporting.
* **Fold Stability**: Visualize the consistency of model performance across cross-validation folds using the stability violin plot.
