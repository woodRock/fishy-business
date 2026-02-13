REIMS Datasets
================

This framework utilizes a series of datasets derived from **Rapid Evaporative Ionization Mass Spectrometry (REIMS)** analysis of seafood samples.

.. note::
   The source code of this project is open-source (MIT), but the dataset itself is **private research data**. Authorized users can download the data using the :ref:`download-data-ref` command.

Data Source and Acquisition
---------------------------

The REIMS data were provided by **AgResearch, New Zealand**, as part of research into quality assurance systems for marine biomass processing.

The analysis utilized a **Laser-Assisted REIMS** setup, coupling a CO2 laser interface to a Xevo G2 XS quadrupole time-of-flight mass spectrometer (Waters Ltd). Samples were received frozen and analyzed within 10 minutes of removal to prevent lipid oxidation.

* **Mode**: Negative ionization (MS1 only).
* **Scan Range**: m/z 50–1200.
* **Resolution**: 2,080 distinct m/z features.

Data Curation and Preprocessing
-------------------------------

Data processing included baseline removal and lock mass correction (using oleic acid, C18:1). Each spectrum comprises **2,080 distinct m/z features**, spanning a range from approximately 77.04 m/z to 999.32 m/z.

Preprocessing includes:
1. **TIC Normalization**: Accounting for sample-to-sample variations in ionization efficiency.
2. **Min-Max Scaling**: Normalizing feature intensities to the range [0, 1].

Analytical Tasks
----------------

The curated data is split into five distinct datasets tailored for specific analytical tasks:

1. Fish Species Identification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Goal**: Distinguish between Hoki and Mackerel.
* **Samples**: 106.
* **Significance**: Fraud prevention and food authentication.

2. Body Part Identification
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Goal**: Identify 7 fish parts (Fillets, Heads, Livers, Skins, Gonads, Guts, Frames).
* **Samples**: 33.
* **Significance**: Process automation and biomass value maximization.

3. Oil Contamination Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Goal**: Predict oil concentration at 7 levels (0% to 50%).
* **Samples**: 126.
* **Significance**: Equipment safety and lubricant detection.

4. Cross-species Adulteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Goal**: Detect if premium species (Hoki) have been diluted with cheaper ones (Mackerel).
* **Samples**: 144.
* **Significance**: Economic fraud detection.

5. Batch Detection (Pairwise)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Goal**: Identify if two samples originate from the same processing batch.
* **Format**: 2,556 pairwise comparisons.
* **Significance**: Traceability and food safety.

Summary Table
-------------

.. list-table:: Summary of Classification Datasets
   :widths: 25 10 10 25 30
   :header-rows: 1

   * - Dataset
     - Examples
     - Features
     - Class Labels
     - Split Type
   * - Fish Species
     - 106
     - 2,080
     - Hoki, Mackerel
     - 5-Fold CV
   * - Fish Body Part
     - 33
     - 2,080
     - 7 Categories
     - 3-Fold CV
   * - Oil Contamination
     - 126
     - 2,080
     - 7 Levels
     - 5-Fold CV
   * - Cross-species
     - 144
     - 2,080
     - Pure/Mixed
     - 5-Fold CV
   * - Batch Detection
     - 2,556
     - 2,080
     - Same/Different
     - 60/20/20 Fixed
