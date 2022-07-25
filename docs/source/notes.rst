Notes
=====

REIMS 
-----

These are my notes from the presentation slides on REIMS https://github.com/woodRock/fishy-business/blob/main/resources/REIMS.pdf

Notes: 
    * REIMS -  Rapid Evaporative Ionisation Mass Spectrometry (REIMS) - a technique that allows rapid high resolution mass spectral fingerprinting of samples with no sample preparation. 
    * Motivation:
        * High resolution masses can be used for basic compound identification. 
        * Existing MS/MS technique is possible, but time-consiming and required preprocessing and manual feature selection. 
    * Method: 
        * Samples taken from frozen (-80°c) tissue.
        * Thawing may lead to better results. 
        * Calibration (1/2 day), cleaning required after each sample. 
        * Takes 8 mins / plate to process a sample. 
        * 96 samples in triplicate, + 9 quality control (QCs), 306 samples total. 
    * Feature Selection: 
        * 2080 features detected, GC-based FS method reduce to 1384 features. 
    * Exploratory Data Analysis (EDA):
        * Unsupvervised Principle Component Analysis (Abdi 2020)
        * Supervised OPLS-DA (Boccard 2013)
    * QC - When quality controlled (QC) samples were only included with RSD > 30% removed, the results were similar, and the data is likely less noisy. 
    * PCA - provides an overview of the variance in the daa. Difficult to gain intuition due to the large number classes. Notably "MG" is seperated. 
    * PC2 - the principal component 2 exmplains the variance between Hoki and Mackeral, "HM" the hoki-mackeral mix, is clearly spread acorss the two clusters. 
    * OPLS-DA is a form of supverised discriminant analysis (DA), similar to PLS-DA, except for forcing the difference between two groups to be along the first component. 
    * OPLS-DA results: 
        * For seperating Hoki and Mackeral - the model is robust (Q2 = 0.9).
        * struggles with Hoki and Hoki-Mackeral Mix - (Q2 = 0.37), after FS with VIP <1 (Q2 = 0.48). 
        * Struggles for Mackerl and Hoki-Mackeral Mix - (Q2 = 0.36), after FS with VIP <1 (0.41). 
    * Summary: 
        1. REIMS can classify Hoki and Mackeral. 
        2. Hoki, Mackeral and Hoki-Mackeral mix is more difficult. 
        3. Samples taken in sequential order, but plates randomized to minimize impact of instrumental drift. 
        4. More steps to combat instrumental drift should be addressed.  
