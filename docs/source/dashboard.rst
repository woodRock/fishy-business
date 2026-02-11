Interactive Dashboard
=====================

The framework includes a comprehensive web-based dashboard built with Streamlit for interactive data exploration, training monitoring, and biomarker discovery.

Launching the Dashboard
-----------------------

To start the dashboard, run the following command from the root of the repository:

.. code-block:: bash

   streamlit run dashboard/app.py

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
* **Global Biomarker Analysis**: Aggregate feature importance across multiple samples to identify stable diagnostic peaks.
* **Class Comparison**: Highlight unique biomarkers in Gold/Silver directly on class-representative spectra to verify their alignment with physical peaks.
