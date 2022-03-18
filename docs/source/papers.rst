Literature Review
=================

akkaya2019solving
-----------------
    * Akkaya et al. propose a robotic hand that "single-handedly" solved a Rubiks cube \cite{akkaya2019solving}. 
    * They use Automatic Domain Randomization (ADR) and simulation to impute data for physics-based problems. 
    * This technique was used to solve a Rubiks cube "single-handedly" by simulation. 
    * It can be difficult to model an accurate physics engine.
    * Instead, ADR solves all possible sets of physics environments within given constraints for the Rubiks cube. 
    * Through simulation, they create a model that generalizes well, with very little real-world experimentation needed.

al2019survey
------------
    * Survey of evolutionary machine learning - Vuw staff. 
    * **TODO** read 

aizerman1964theoretical
-----------------------
    * The original hyperplane algorithm used a linear kernel.

bi2020gc
--------
    * Bi et al. proposed a CNN model that incorporated GC-MS data fusion for food science.
    * The high-dimensional data was naturally suited towards the CNN.
    * Their work classified the flavour quality of peanut oil with 93\% accuracy.
    * Similar to this project, the existing technique for analysis was intractable large scale.
    * The fusion of existing datasets improved the efficacy of their model.

boser1992training
-----------------
    * Kernal trick for SVM.
    * These employ the kernel trick.

liu1995chi2
-----------
    * chi^2 classifier.     

craik1972levels
---------------
    * Levels of processing: A framework for memory research. 
    * Elaborative rehearsal requires deeper processing than maintainence rehearsal. 

craik1975depth
---------------
    * Deeper processing, semantic over structural or phonetic, better. 
    * Depth processing increased later recognition of words in a list. 
    * Annecodte, study: skim-read vs. thoughtful reading. 

chase1973perception
-------------------
    * Domain expertise allows people to build meaningful schema to represent patterns.
    * Expert chess players recall 16 pieces, intermeidate 8, novice 4 when arranged in meaninful positions. 
    * Recall was consistant for levels of expertise on nonsense chess boards. 
    * Our mental schemas for encoding patterns break on noise (unseen data). 

cortes1995support
-----------------
    * Cortes and Vapnik proposed the Support Vector Machine (SVM).
    * This model creates a hyperplane that can draw distinct class boundaries between classes.
    * We call these class boundaries the support vectors.
    * We are performing multi-class classification, so it used a one-vs-all approach \cite{sklearn2021feature}.
    * This creates a divide between one class and the rest, then repeats for the other classes.

da2018evolutionary
------------------
    * Evolutionary Computation Approaches to Web Service Composition. 
    * Service composition is an NP-hard combinatorial problem - local search via heuristic is needed. 
    * Optimizes fitness as multi-objective function of correctness and exectution time. 
    * Graph building algorithm that uses evolutionary techniques, mutation and crossover. 
    * Don't reinvet the wheel, encourage reuse of existing services. 

ding2005minimum
---------------
    * Minimum Redundancy - Maximum Relevance (MRMR)

eder1995gas
-----------
    * Gas chromatography (GC) \cite{eder1995gas} is a method that can identify chemicial structures in these fish oils.
    * This produces high-dimensional low sample size data from the fish oils.
    * Chemists compare a given sample to a reference sample to determine what chemicals are present.
    * The existing analytical techniques to perform these tasks are time-consuming and laborious.

fix1989discriminatory
---------------------
    * K-nearest neighbours (KNN).

fukushima1982neocognitron
-------------------------
    * Rectified Linear Unit (ReLu) paper. 
    * Activation function for neural networks. 
    * Shares nice properties of linear function. 
    * But allows for non-linearities to be captured. 

hand2001idiot
-------------
    * Naive bayes. 

ho1995random
-------------
    * Random forest. 

karras2020analyzing
-------------------
    * StyleGAN 
    * Latent layer representation. 
    * Manipulating latent layer gives a sense of semantically meaninful feature space. 
    * We can see the change in style that sampling latent layer gives. 

kennedy1995particle
-------------------
    * Original PSO algorithm.

kennedy1997discrete
-------------------
    * PSO for feature selection. 

kingma2014adam
--------------
    * Adam optimizer for neural networks. 

koppen2000curse
---------------
    * Curse of dimensionality. 

kullback1951information
-----------------------
    * Kullback-Leibler (KL) divergence. 
    * Measures distance between two probability distributions. 
    * Most common loss function for deep learning with stochastic gradient descent. 

lecun1989generalization
-----------------------
    * Original Convolutional Neural Network (CNN) paper. 

loh2011classification
---------------------
    * Decision tree. 

marhsall2022cybermarine
-----------------------
    * Cybermarine research magazine aims. 
    * Focus on reducing by-product. 
    * Non-destructure methods for analysis of chemical compounds in fish oil. 
    * Factory of the future - uses AI to inform decisions in the assembly line.

matyushin2020gas
----------------
    * Matyshuin et al. proposed a stacking model for analysis of gas-chromatograph data.
    * It stacked the results of 1DConv, 2DConv, Deep Residual MLP and XGBoost.
    * Their model predicted the retention index for samples.
    * A retention index is a standardized value that only depends on the chemical structure of a compound.
    * Once identified the retention index can be used for further identification.
    * GC-MS data has underlying patterns that correspond to chemical compounds.

mikolov2013linguistic
---------------------
    * Mikolov et al. found the word embeddings used in NLP were semantically meaningful \cite{mikolov2013linguistic}. 
    * They showed arithmetic could be applied to these word vectors that were interpretable. 
    * For example "King" - "Man" + "Woman" = "Queen". 
    * The feature space was semantically meaningful, which serves as a powerful representation, that we intuitively reason with. 
    * Similar thought has been applied to computer vision \cite{olah2018building, karras2020analyzing}. 
    * Semantically meaningful feature spaces allow for intuition about the behaviour of complex models, be it through visualisation or arithmetic.

nguyen2014filter
----------------
    * Nguyen et al. proposed a wrapper based PSO technique for feature selection in classification.
    * The algorithm uses a wrapper based fitness function of the classification error rate.
    * The local search only considers the global best using a filter based method.
    * It draws from the strengths of filter and wrapper based feature selection.
    * This proposed method outperformed three state-of-the-art and two traditional feature selection methods.

olah2018building
----------------
    * Semantically meaningful features in computer vision. 
    * Distill https://distill.pub/2018/building-blocks/
    * Visualization techniques are powerful for understanding black-box systems.
    * Gain intution for semantically meaninful features in complex models. 

raine1997brain
--------------
    * Muderers pleading not guilty be reason of insanity (NGRI).
    * Pre-disposition to less activity in their pre-frontal cortex. 
    * Pre-frontal cortex associated with goal-directed planning and delayed gratification. 
    * Different brain chemistry meant more likely to perform violent impulsive behaviour. 
    * Justification for lebotomy - electrocution of pre-frontal cortex - now replaced by anti-psychotics. 

restek2018high
--------------
    * Explanation of gas-chromatraphy in food science for FAMEs. 

robinson2020genetic
-------------------
    * Demelza et al. proposed a feature and latent variable selection method for regression models in food science.
    * The vibrational spectroscopy dataset shared similarities in its high dimensionality and food science domain.
    * The purposes GA-PLSR generalized better and produced fewer complex models.
    * The study showed that Genetic Algorithms are powerful tools for feature selection in food science.

robnik2003theoretical
---------------------
    * releifF classifier. 

scholkopf2000new
----------------
    * Nu-SVC classifier. 
    * Setting the number of support vectors is a hyper-parameter.
    * Usually this is learned by the system. 

tegmark2020ai
-------------
    * Tegmark et al. developed they AI Feynman \cite{udrescu2020ai}. 
    * This algorithm can derive physics equations from data using symbolic regression. 
    * Symbolic regression is a difficult task, but by simplifying properties exhibited by physics equations (i.e symmetry, composability, separability), the problem can be reduced. 
    * Their work uses blackbox neural networks, to derive interpretable models that can easily be verified by humans. 

tomasi2004correlation
---------------------
    * Tomasi et al. investigated correlation optimisation warping (COW) and dynamic time warping (DT) for preprocessing chromatography data.
    * Unconstrained dynamic time warping was found to be too flexible. 
    * The algorithm overcompensated when trying to fix the alignment in the data.

tran2018variable
----------------
    * Tran et al. propose a Variable-Length PSO.
    * Traditional PSO methods for feature selection are limited in the fixed length of their representation.
    * This leads to both high memory usage and computational cost.
    * The proposed algorithm allows particles to have shorter and different variable lengths.
    * Their length changing mechanism allows PSO to escape local optima.
    * Results across several high dimensional datasets showed improved performance in terms of computational time, fewer features selected and classification accuracy.

wolpert1997no
-------------
    * No free lunch theorum. 
    * No classification algorithm that beats the rest for every problem. 
    * As training instances approaches infinity, classification accuracy on all distributions of noise, approaches predicting mean class. 
    * All machine learning algorithms are task specific, don't generalize to all problems, no artifical general intelligence (AGI), yet... 

xue2014particle
---------------
    * Brown et al. proposed a PSO with novel initialising and updating mechanisms.
    * The initialization strategy utilized both forward and backwards selection.
    * The updating mechanism overcame the limitations of the traditional method by considering the number of features.
    * The proposed algorithm had better performance in terms of computing, fewer features selected and classification accuracy.

zhang2008two
------------
    * Zhang et al. proposed a 2-D COW algorithm for aligning gas chromatography and mass spectrometry. 
    * The algorithm warps local regions of the data to maximise the correlation with known reference samples. 
    * This work uses data fusion with labelled reference samples, to improve the quality of new samples.
