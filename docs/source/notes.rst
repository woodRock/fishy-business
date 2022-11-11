Notes
=====

2022-08-23 - REIMS 
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

Related:
    * See 2022-08-23 - PFR Daniel and Kevin

ECS Grid 
--------

Commands: 
    - To submit a job run ``qsub job.sh`` where ``job.sh`` is the name of the script.
    - To see the state of current jobs run ``qstat``.
    - To see why a script is in an error state run ``qstat -explain E -j <jobid>``. 

Notes: 
    - Instructions for the grid can be found here https://ecs.wgtn.ac.nz/Support/TechNoteEcsGrid#A_basic_job_submission_script 
    - Remote access to the grid is available via SSH https://ecs.wgtn.ac.nz/Support/TechNoteWorkingFromHome 
    - My griduser folder is ``woodjess3``, but my ECS username is ``woodj4``, be careful with this difference when writing scripts. 
    - For email notifications of job completion, I must use my ECS email Jesse.Wood@ecs.vuw.ac.nz. 

2022-11-11 Differential Equations
---------------------------------

"Any equation that caontains derivities [...] is called a differential equation."

Notes: 
    * A differential equation is an queation that contains a derivite. 
    * Examples of differential equations include Newton's second law, hookes law (or the spring equation). 
    * Newton's second law states that a froce is equal to the mass of an object multiplied by its acceleration, :math:`F = ma`
    * We can express accelaration as the first-order derivite of velocity :math:`\frac{d}{dt}(v)`.
    * Therefore we can give Newton's second as, :math:`f = \frac{d}{dt}(mv)`.
    * This is an example of a differential equation (DE). 
    * Hookes law, which can be derived from newtons first law (describing inertia) can be given as, 
    :math:`x''(t)=\frac{k}{m} x(t)` 
    or 
    :math:`F = kx`
    * That is the second-order derivitive can be expressed as a function of itself multiplied by a constant. 

Bias Variance Tradeoff 
----------------------
