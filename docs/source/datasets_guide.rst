Dataset Configuration
=====================

This guide explains how to integrate new spectral datasets into the framework.

Data Format
-----------
The framework primarily expects spectral data in **Excel (.xlsx)** or **CSV** format.
The standard structure used in this project is:

*   **Row 1**: Header row.
*   **Column 1**: Identifier or Target (e.g., Species name, or specific identifier).
*   **Columns 2+**: Features (typically intensity values for specific m/z values).

Registering a Dataset
---------------------
To add a dataset, edit ``fishy/configs/datasets.yaml``. Each entry supports several keys:

.. code-block:: yaml

   my-new-dataset:
     filter_rules:
       exclude_mz: ["QC"]  # Drops rows containing "QC" in the m/z column
       include_mz_pattern: "Hoki" # Only keeps rows matching this pattern
     label_encoding:
       type: "sklearn"  # Supported: sklearn, one_hot, map, regex_float

Encoding Types
--------------
*   ``sklearn``: Uses a standard LabelEncoder on the first column.
*   ``one_hot``: Converts categorical names into binary vectors.
*   ``map``: A manual dictionary mapping substrings to label vectors.
*   ``regex_float``: Extracts a numeric value from a string using regex (useful for regression).

Filtering Logic
---------------
The ``DataProcessor`` automatically filters out "QC" (Quality Control) samples by default. You can add custom rules to ignore specific batches or experimental conditions without modifying the raw data files.
