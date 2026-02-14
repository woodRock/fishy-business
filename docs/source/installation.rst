Installation
============

Prerequisites
-------------
- Python 3.9 or higher

Install from Source
-------------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/woodRock/fishy-business.git
   cd fishy-business

2. Install dependencies:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .

.. _download-data-ref:

3. Download the Private Dataset:
--------------------------------

The REIMS dataset is private. To download it, you will need a GitHub Personal Access Token (PAT):

.. code-block:: bash

   fishy download-data --token <YOUR_GITHUB_TOKEN>

*Tip: You can also set the FISHY_DATA_TOKEN environment variable.*

4. (Optional) Install development dependencies for testing:

.. code-block:: bash

   pip install pytest pytest-cov pytest-mock
