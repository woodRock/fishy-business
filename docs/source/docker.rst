🐳 Docker Integration
====================

For maximum reproducibility and ease of deployment, Fishy Business can be run entirely within a Docker container. This ensures that all dependencies (including complex C-libraries for spectral analysis) are correctly configured.

Building the Image
------------------

From the root of the project directory, build the Docker image:

.. code-block:: bash

   docker build -t fishy-business .

Running the Dashboard
---------------------

The default behavior of the container is to launch the Streamlit dashboard on port ``8501``.

Basic Run
~~~~~~~~~

.. code-block:: bash

   docker run -p 8501:8501 fishy-business

With Persistent Data
~~~~~~~~~~~~~~~~~~~~

To ensure your training results and datasets persist outside the container, mount your local ``data`` and ``outputs`` directories:

.. code-block:: bash

   docker run -p 8501:8501 
     -v $(pwd)/data:/app/data 
     -v $(pwd)/outputs:/app/outputs 
     fishy-business

Using the CLI via Docker
------------------------

You can use the container to run training experiments or manage data without installing Python locally.

Download Data
~~~~~~~~~~~~~

.. code-block:: bash

   docker run -it -v $(pwd)/data:/app/data fishy-business 
     python -m fishy download-data --token <YOUR_GITHUB_TOKEN>

Run a Training Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker run -it 
     -v $(pwd)/data:/app/data 
     -v $(pwd)/outputs:/app/outputs 
     fishy-business 
     python main.py train -m transformer -d species --epochs 50

Weights & Biases Integration
----------------------------

To log results to W&B from within Docker, pass your API key as an environment variable:

.. code-block:: bash

   docker run -p 8501:8501 
     -e WANDB_API_KEY=<your_key> 
     -v $(pwd)/outputs:/app/outputs 
     fishy-business

Summary
-------

Using Docker is highly recommended for:
* **Server Deployment:** Hosting the dashboard on a remote research server.
* **Reproducibility:** Ensuring your thesis results aren't affected by local library updates.
* **Quick Setup:** Skipping the installation of complex dependencies like ``XGBoost`` or ``Graphviz``.
