.. _literature-review:

Literature Review
=================

abdi2010principal
-----------------
Principal component analysis 

(Abdi 2010) propose Principal Component Analysis (PCA) for dimensionality reduction. 

Method: 
    * Project data along the principal components, the axis of maximum variance in descending order. 
    * The first principal component is the axis of maximum variance, the second principal component is orthogonal to the first and has the second largest variance, and so on.

Related: 
    * :ref:`(Black 2019) <black2019rapid>` uses PCA for preprocessing. 
    * :ref:`(Goodfellow 2016) <goodfellow2016deep>` gives dervation using Linear Algebra. 

adebayo2018sanity
-----------------
Sanity checks for saliency maps

(Adebayo 2018) suggests salience maps are glorified edge detectors.

agarwal2011building
------------------
Building rome in a day

TODO:
    * READ https://dl.acm.org/doi/abs/10.1145/2001269.2001293

aizerman1964theoretical
-----------------------
    * The original hyperplane algorithm used a linear kernel.

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

ba2016layer
-----------
Layer Normalization

(Ba 2016) propose Layer Normalization, a regularization technique for deep learning.

Available: https://arxiv.org/abs/1607.06450

Related: 
    * Transformer :ref:`(Vaswani 2017) <vaswani2017attention>`

balas1969machine
----------------
Machine Sequencing Via Disjunctive Graphs: An Implicit Enumeration Algorithm

(Balas 1969) proposes a disjunctive graph for machine sequencing.

Available: https://pubsonline.informs.org/doi/abs/10.1287/opre.17.6.941

Related: 
    * Mentioned in :ref:`2023-10-06 - ECRG <2023-10-06 - ECRG>`
    * Vehicle routing with transformers :ref:`(Kool 2018) <kool2018attention>`
    * node2vec :ref:`(Grover 2016) <grover2016node2vec>`

bao2022estimating
-----------------
    * Estimating the Optimal Covariance with Imperfect Mean in Diffusion Probabilistic Models 
    * Diffusion Probabilistic Models (DPM) are special Markov Models with Gaussian Transitions. 
    * Paper shows how to go from noisy-to-clean with a deterministic process. 
    * A new approach to diffusion based models.

batstone1999new
---------------
New Zealand’s quota management system: the first ten years

(Batstone 1999) describes the first 10 years of the New Zealand Quota Management System (QMS) for fisheries management.

Related: 
    * :ref:`(lock2007new) <lock2007new>` gives a history of NZ QMS for first 20 years.

banzhaf2006artificial
---------------------
From artificial evolution to computational evolution: a research agenda

(Banzahf 2006) proposes a research agenda for evolutional computataion, published in nature magazine. Available https://www.nature.com/articles/nrg1921

TODO [ ] Read this paper! 

Related: 
    * See ECRG - 2022-10-28

banzhaf2009genetic
------------------
    * Genetic Programming: An Introduction On The Automatic Evolution Of Computer Programs And Its Applications
    * TODO [ ] must read book for foundations of GP. (buy?)

behmo2010towards
----------------
Towards optimal naive bayes nearest neighborhood

`(Behmo 2010) <https://link.springer.com/chapter/10.1007/978-3-642-15561-1_13>` proposes a Naive Bayes Nearest Neighbour (NBNN) classifier.

Related: 
    * :ref:`(Crall 2013) <crall2013hotspotter>` uses LBNN for instance recognition. 
    * :ref:`(McCann 2012) <mccann2012local>` proposed Local Naive Bayes Nearest Neighbour (LNBNN).

bengio2017consciousness
-----------------------
    * The consciousness prior

bi2020gc
--------
GC-MS Fingerprints Profiling Using Machine Learning Models for Food Flavor Prediction 

(Bi 2022) proposed a CNN model that incorporated GC-MS data fusion for food science.

Data: 
    * Food flavour quality evaluation is interesting, but lacks evaluation techniques. 
    * Olfactometry, an instrument used to detect and measure odor dilution, is unreliable due to user error or systematic laboratroy effect. 
    * Existing technique for analysis was intractable large scale.
    * Evaluated on existing Gas Chromatography - Mass Spectrometry (GC-MS) measurements on peanut oil data.
Method: 
    * A novel fingerprint modelling and profilling process 
    * Dataset expansion 
Results:
    * Their work classified the flavour quality of peanut oil with 93\% accuracy.
    * Dataset expansion: the fusion of existing datasets improved the efficacy of their model.
Why it matters? 
    * CNN can make accurate predictions on high-dimensional GC-MS data. 
    * Proposes method can automate aroma analysis, reducing human labour, and improving accuracy.

Related: 
    * :ref:`(Eder 1995) <eder1995gas>` is the original gas chromatrogaphy (GC) paper. 
    * :ref:`(Zhang 2008) <zhang2008two>` preprocssing method for aligning gas chromatography (GC).
    * :ref:`(Wood 2022) <wood2022automated>` performs classification / feature selection on gas chromatography data. 

bifet2007learning
-----------------
Learning from time-changing data with adaptive windowing

(Bifet 2007) propsoed the ADWIN method for detecting concept drift in data streams.

Related: 
    * See (:ref:`Gomes 2020<gomes2020ensemble>`) for paper that cites. 
    * See :ref:`2023-02-16 - FASLIP<2023-02-16 - FASLIP>`

black2017real
---------------
A real time metabolomic profiling approach to detecting fish fraud using rapid evaporative ionisation mass spectrometry

(Black 2017) prose REIMS for fish fraud detection.

Notes: 
    * TODO [ ] Read this paper 

Related: 
    * :ref:`(Black 2019) <black2019rapid>` propose REIMS for rapid and specific identification of foffal cuts within minced beef samples.
    * :ref:`(Wood 2022) <wood2022automated>` performs classification / feature selection on gas chromatography data on fish data. 

black2019rapid
--------------
Rapid detection and specific identification of offals within minced beef samples utilising ambient mass spectrometry

(Black 2019) propose REIMS for rapid and specific identification of foffal cuts within minced beef samples. 
    
Background: 
    * Criminals add stuff to meat products (adulteration) for economic gains. 
    * Meat adulteration in non-meat products of <1% expected (and allowed) as it is considered cross-contaminiation, and not for economic gains. 
    * Adulterations levels from (15%-20%) are considered criminal as they are likely for economic gains.
    * 2013 European Horsemeat scandal is an example of this. 
    * In repsonse, European Union (EU) decalared that non-meat opffcal cuts must be declared on product labels. 
    * Recent study (BBC 2018) in the UK (n=665), found >1/5 of samples contained non-declared meat species.
    * E.g., for 2013 European horsemeat scandal, REIMS could detect the adulteration, and identify that adulterant as horse.
    * Rapid evaportive ionization mass spectrometry (REIMS)
    * Minced beef products are often ready-to-go, and pre-cooked, so a method is needed that works on raw/cooked meat products. 

Motivation: 
    * DNA sequencing can only differentiate between different species, not offal adulteration from the same species. 
    * Virbration spectroscopy cand etect adulteration, but not the specific offal present. 
    * Both DNA methodologies and vibrational spectroscopy are ineffective at detecting these adulterations. 
    * Traditional chromatroagprahy/mass spectromety hasn't been tried, due to time to prepare/analyze samples. 
    * Ambient Mass Spectromerty (AMS) has potential to identify unique/signficiant metabolites. GC-MS cannot do this!
    * Significant Markers (or important variables) are ions that are unique to a specific offal cut, and present in all samples. 
    * Looking for a reliable, accurate and rapid method that can be deployed in a food processing plant for quality assurance. 
    * Looking for a model that can detect adulteration levels for criminal activity adulteration for economic gains.

Data: 
    * Cheap offal products can be addded to beef tissues when they are minced in food processing to cut corners and increase profits.
    * Minced beef (1 class) with alteration from beef brain, heart, kidney, large intestine and liver tissues (5 classes).
    * Outliers are hybrid spectra - a homogenous mix of beef and adulteration - at a given adulteration level (i.e. 20%, 10%, 5%, 1%). 
    * Pre-processing (before PCA-LDA):
        1. Prototpye abstract model builder 
        2. Masslynx pre-processing algorithms
        3. Background subtracted 
        4. Lockmass corrected 
        5. Normalized by TIC (total ion count) 
    * Post-processing (after PCA-LDA): 
        1. Mean-centered 
        2. Pareto scaled 
        3. Grouped by class 
    * Method facilitates real-time classification, with classification output prodived every second. 
    * METLIN metabolies databas, and LIPID MAPS can proved annotated lables for spectra. 

Method: 
    * They propose REIMS for detecting beef adulteration.
    * Metrics: 
        1. :math:`R^2` measures the variation in samples. 
        2. :math:`Q^2` measures the accuracy of classification of class. 
        3. RMSE-CV measure cross validated root means squared error. 
    * Feature Selection: 
        * Variable Importance Projection (VIP)
        * S-plots? 
    * Chemometric analysis (VIP + S-plots) of REIMS could detect unique/significant markers. 
    * Prinicapl component anaylsis linear discriminat anaylsis (PCA-LDA) (Abdi 2010) using orthogonal partial least squares discriminant analysis (OPLS-DA) (Boccard 2013).
    * PCA-LCA used for dimensionality reduction - classification, respectively. 
    * Detect outliers based on standard deviation outside 20 :math:`\sigma` of the mean for any class. 
    * They provide a very detailed description of their method from the chemistry side, including instruments and their settings. Good for reproducability and understanding.

Results: 
    * PCA/LDA (with manual hyper-parameter tuining) can effecitvely detect adulteration - i.e. cluster different classes within adulteration levels (i.e. 15-20%).
    * The adulteration levels were measured on raw/boiled minced beefs. 
    * Raw: brain (5%), heart (1-10%), kidney (1-5%), large intestincce (1-10%), liver (5-10%).
    * Beef and large intestine were too similar to detect outliers with PCA-LDA. Perhaps very similar tissue composition.
    * Within adulteration levels (i.e. 15-20%), their model can predict adulteration with perfect precision :math:`P(C|\hat{C}) = 1`, i.e., all predicted alduterations were correct.
    * Boiled: brain (5-10%), heart (1-10%), kidney (1-5%), large intestine (1-10%), live (5-10%). 
    * Boiled samples are harder to classify. More principle components were needed to correctly identify adularation for boiled samples. 

Why it matters? 
    * REIMS is a cheap and rapid method for detecting adulteration in minced beef in a factory setting. 
    * REIMS can detect both adulterations, and the specific adulteration present, superior to other methods.
    * Many meat products are pre-cooked, REIMS detects adulteration (at criminal levels) in raw/boiled meat. 
    * REIMS can provide a paradigm shift across many authenticity applications.  
    * (Black 2017) shows can be successfully applied to fish REIMS data.

Limitations: 
    * Basic dimensioanlity reduction techniques (PCA) were used. Future work should consider t-SNE. 
    * Basic sueprvised statistical models were (LDA, OPLS-DA) were used for classification. Future work should consider GANs, VAEs, Diffusion, CNNs. 
    * Potential for transfer learning (encorporate previously existing data) to improve performance for few-shot classification tasks. 

Related: 
    * :ref:`(Black 2017) <black2017real>` use REIMS for fish fraud detection. 
    * (BBC 2018) Recent study in the UK (n-665), found >1/5 of samples contained non-declared meat species. https://www.bbc.com/news/uk-45371852


blattmann2023align
------------------
Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models

Notes: 
    * NVIDIA Paper on text-to-video synthesis.
    * [Available] https://arxiv.org/abs/2304.08818
    * TODO [ ] Read

Background: 

Motivations: 

Data: 

Method: 

Results: 

Why it matters? 

Limitations:

Related: 
    * DDPM :ref:`(Ho 2020) <ho2020denoising>` was the original Denoising diffusion probabilistic models (DDPM)
    * DDIM :ref:`(Song 2020) <song2020denoising>` Denoising diffusion implicit models (DDIM), improved DDPM
    * Elucidating :ref:`(Karras 2022) <karras2022elucidating>` provided a concrete design space for LDM architectures. 

boccard2013consensus
--------------------
A consensus orthogonal partial least squares discriminant analysis (OPLS-DA) strategy for multiblock Omics data fusion

Notes: 
    * TODO [ ] Read 

Related: 
    * :ref:`(Black 2019) <black2019rapid>` use OPLS-DA for adulteration detection in minced beef.
    * :ref:`(Black 2017) <black2017real>` uses OPLS-DA for fish fraud detection. 

bourque2018ten
--------------
Ten things you should know about transposable elements

Related: 
    * Julie discussed this at ECRG - 2022-10-14 
    * :ref:`(Hof 2016) <hof2016industrial>` gives an example of tranposons affecting moths. 
    * :ref:`(Kulasekara 2014) <kulasekara2014transposon>` says changes passed to offspring. 

boser1992training
-----------------
    * Kernal trick for SVM.
    * These employ the kernel trick. 

breiman2017classification
-------------------------
Classification and Regression Trees 

(Breiman 2017) is the book on CART.

Available: https://www.taylorfrancis.com/books/mono/10.1201/9781315139470/classification-regression-trees-leo-breiman-jerome-friedman-olshen-charles-stone

Background: 
    * Book orginally published in 1984
    * decision trees are an algorithm that only contains conditional control statements, i.e. if-else statements.
    * The acronym is Classification and Regression Trees (CART).
    * In 1984, Breiman et al \cite{breiman2017classification} proposed . 
    The classification task predicts the class label an instance is most likely to belong to. 
    
Representation: 
    * CART uses a tree-based structure of both nodes, branches and leaves. 
    * The nodes are where decisions are made, branches give the outcome of those decisions, and leaves give the predicted class label.

Method: 
    * The algorithm uses a greedy approach to build the tree.
    * It evaluates all possible splits and selects the one that best reduces the impurity of the resulting subsets.
    * It splits at the feature that is the best splitting point.
    * CART continues to split until a stopping rule is met, or no further best splits are available.
    * For classification, Gini impurity is the splitting criterion. 
    * The lower the Gini impurity, the more pure a subset is. 
    * For regression, the residual reduction is the splitting criterion. 
    * The lower the residual reduction, the better fit the model is to the data. 
    * Pruning: to prevent overfitting of the data, pruning can remove branches that do not improve the model's performance.
    * Cost complexity and information gain pruning are two popular techniques.

Applications: 
.. epigraph::
    In 1977-1978 and again in 1981, the EPA funded projects for the construction of classification trees to recognize the presence of certain elements in compounds through the examination of their mass spectra. The EPA, as part of its regulatory function, collects numerous samples of air and water containing unknown cdompounds and tries to determine the presence of toxic substances. According to McLafferty: "The fragment ions indicate the pieces of which the molecule is composed, and the interpreter attempts to deduce how these pieces fit together in the original molecular structure. In such correlations have been achieved for the spectra of a variety of complex molecules." The critical element of the bromine tree was the construction of a set of questions designed to recognize bromine hallmarks. If bromine occurs in combination with chlorine, then since chlorine (weight 35) has an isotope of weight 37 that occurs 24.5 percent of the time, there is a different theoretical ratio vector.

    -- Mass Spectra Classification

Related: 
    * :ref:`(Von 1986) <von1986decision>` is another decision tree paper from the 80s.
    * Geeks for Geeks https://www.geeksforgeeks.org/cart-classification-and-regression-tree-in-machine-learning/

brewer2006brown
---------------
    * Flashbuld memories - recollections that seem vivid and clear, so we take them to be accurate. 
    * Most likely occur for distinct stronly positive or negative emotional events. 
    * Weddings, Funerals, Deaths, Tragedy, Violence. 
    * We are more likely to be confident these are correct.
    * But our memory is shit, so we often re-write and incorrectly recall these events. 
    * The distinictness of flashbulb memories, does help recall them longer, but does not guarantee correctness. 

bridle1989training
------------------
Training stochastic model recognition algorithms as networks can lead to maximum mutual information estimation of parameters

(Bridle 1989) is the first paper to mention "softmax" in neural networks.

Related: 
    * According to StackExchange, this is the original "softmax" paper for neural networks https://ai.stackexchange.com/questions/22426/which-paper-introduced-the-term-softmax

brochu2010tutorial
------------------
A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning

(Brochu 2010) is useful for Gaussian Processes, predictions with confidence intervals, or uncertainty thresholds.

Notes: 
    * A Tutorial on Bayesian Optimization of Expensive Cost Functions
    * Application: 
        1. Active User Modeling 
        2. Hierarchical Reinforcement Learning
    * Covers the theory and intuition behind Bayesian optimization
    
bromley1993signatured
---------------------
Signature verification using a" siamese" time delay neural network

(Bromley 1993), from LeCun's lab, proposes Siamese Neural Networks, a contrastive learning technique, for signature verification.

Task: 
    * Signature verification
    * Pair-wise comparison of signatures.
    * Given: 
        * Reference - a genuine signature 
        * Query - a signature to be verified.
    * Determine if query is a genuine signature

Data:  
    * Signature verification.
    * Eliminate redundancies - forgeries must attempt to copy a genuine signature.
    * Genuine signatures have between 80% to 120% of the original strokes of the reference signature.
    * Note: 120% implies a signature with a few more strokes than the reference is still considered genuine.
    * 219 people signed between 10 and 20 signatures each, 145 signed genuines, 74 signed forgeries. 
    * Few-shot learning - A person must have signed at least 6 genuine signatures or forgeries. 

Method:
    * Siamese network - two identical networks, with shared weights.
    * The two networks are fed the reference and query signatures.
    * Euclidean distance between the two networks is used to determine if the query is genuine.
    * A form of contrastive learning. 

Results: 
    * Best performance was obtained with Network 4. With the threshold set to detect 80% of forgeries, 95.5% of genuine signatures were detected (24 signatures rejected).
    * Performance could be improved to 97.0% genuine signatures detected (13 rejected) by removing all first and second signature from the test set 2. 
    * For 9 of the remaining 13 rejected signatures pen up trajectories differed from the person's typical signature.

Why it matters? 
    * Siamese networks are a form of contrastive learning.
    * Contrastive learning is a form of self-supervised learning.
    * Contastrive learning is an efficient technique for few-shot learning.

Limitations: 
    * "Another cause of error came from a few people who seemed unable to sign consistently and would miss out letters or add new strokes to their signature."
    * The authors note that the performance of the system is limited by the quality of the signatures. 

Applications: 
    * (Bromley 1993) was a proof-of-concept for the signature verification system.
    * It worked equally well for American, European and Chinese signatures. 
    * A field trial needed before it could be deployed in a real-world setting.

Related: 
    * :ref:`(Zhu 2020) <zhu2020masked>` uses Siamese networks for malware detection. 
    * :ref:`(Jing 2020) <jing2022masked>` propose masked siamese networks. 

brosnan2003monkeys
------------------
Monkeys reject unequal pay

(Brosnan 2003), in parntership with Frans de Waal, show that monkeys reject unequal pay.

Notes: 
    * Monkeys are given a simple task with a reward.
    * One monkey is given plain cucumbers, the other is given grapes.
    * The monkey that is given cucumbers goes bananas over the inequity.
    * Repeat experiments where both monkeys are given cucumbers, show no reaction.

Related: 
    * :ref:`(Lex 2022) <lex2022noam>` fairness lead to self-destructive behaviour for retribution in the game of diplomacy.
    * :ref:`(Brown 2022) <brown2022human>` shows that AI can beat humans at diplomacy.

brown2012conditional
--------------------
    * Conditional likelihood maximisation: a unifying framework for information theoretic feature selection
    * Generalized model for information based feature selection methods. 
    * These models generazlize to iterative maximizers of conditional likelihood. 

brown2018superhuman
-------------------
Superhuman AI for heads-up no-limit poker: Libratus beats top professionals

(Brown 2018) shows that AI can beat humans at poker.

Libratus: Brown was also a lead researcher on the Libratus project, which developed an AI system that was able to consistently beat human professionals at two-player no-limit Texas hold 'em poker. 

The research paper describing Libratus was published in the journal Science in 2017 and can be found here: https://www.science.org/doi/full/10.1126/science.aao1733

Related:
    * :ref:`(Lex 2022) <lex2022noam>` interviews Noam Brown, the author of this paper.
    * :ref:`(Brown 2019) <brown2019superhuman>` shows that AI can beat humans at poker.
    * :ref:`(Brown 2022) <brown2022human>` shows that AI can beat humans at diplomacy.
    * :ref:`(Morvavvcik 2017) <moravvcik2017deepstack>` DeepStack beats humans at heads-up no-limit Texas hold 'em poker.

brown2019superhuman
-------------------
Superhuman AI for multiplayer poker

(Brown 2019) shows that AI can beat humans at poker.

Brown was one of the lead researchers on the Pluribus project, which developed a new type of AI system that was able to consistently beat human professionals at six-player no-limit Texas hold 'em poker. 

The research paper describing Pluribus was published in the journal Science in 2019 and can be found here: https://www.science.org/doi/full/10.1126/science.aay2400

Related: 
    * :ref:`(Lex 2022) <lex2022noam>` interviews Noam Brown, the author of this paper.
    * :ref:`(Brown 2018) <brown2018superhuman>` shows that AI can beat humans at poker.
    * :ref:`(Brown 2022) <brown2022human>` shows that AI can beat humans at diplomacy.
    * :ref:`(Morvavvcik 2017) <moravvcik2017deepstack>` DeepStack beats humans at heads-up no-limit Texas hold 'em poker.

brown2020language
-----------------
Language Models are Few-Shot Learners

Notes: 
    * Scaling up language models greatly improves task-agnostic, few-shot performance
    * tasks: NLP datasets, including translation, question-answering, and cloze tasks
    * tasks with on-the-fly reasoning or domain adaptation: unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic.
    * GPT can produce convincing fake news articles that humans struggle to spot.

Related: 
    * :ref:`(Dong 2022) <dong2022survey>` suvery paper on ICL 
    * :ref:`2023-02-22 - Deep Learning  <2023-02-22 - Deep Learning >` discusses this. 

brown2022human
--------------
Human-level play in the game of Diplomacy by combining language models with strategic reasoning.

(Brown 2022) shows that AI can beat humans at diplomacy.

Cicero: Brown co-created an AI system that can strategically out-negotiate humans using natural language in a popular board game called diplomacy which is a war game that emphasizes negotiation.

The research paper describing Pluribus was published in the journal Science in 2019 and can be found here: https://www.science.org/doi/10.1126/science.ade9097

Related: 
    * :ref:`(Lex 2022) <lex2022noam>` interviews Noam Brown, the author of this paper.
    * :ref:`(Brown 2018) <brown2018superhuman>` shows that AI can beat humans at poker.
    * :ref:`(Brown 2019) <brown2019superhuman>` shows that AI can beat humans at poker.
    * :ref:`(Brosnan 2003) <brosnan2003monkeys>` shows monkeys reject unequal pay.

brownlee2016gentle
----------------------
Gentle Introduction to the Bias-Variance Trade-Off in Machine Learning

(Brownlee 2016) shows "[s]upervised learning can be best understood through the lens of the bias-variance tradeoff." 

Available here https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/

Notes:
    * The goal of supervised learning is to find the best estimate function (:math:`f`) for the output variable (:math:`y`) given the input data (:math:`x`) - often referred to as the target function. 
    * Bias are simplfying assumtions made by the model to make the target function easier to learn. 
    * Bias E.g.
        * Low-bias: DT, KNN, SVM 
        * High-bias: LDA, Linear/Logistic Regression 
    * Variance is the amount that the estiamte of the target function will change if different training data were used. 
    * Variance E.g.
        * Low-variance: LDA, Linear/Logistic Regression
        * High-variance: DT, KNN, SVM
    * Trend (often):
        * Linear models will have high-bias low-variance 
        * Non-linear models will have low-bias high-variance 
    * Trade-off E.g.
        * The KNN has low-bais high-variance, tradeoff can be changed by increasing :math:`k` (which increases the number of neighbors that contribute t the prediction), increases the bias of the model. 
        * The SVM has low-bias high-variance, increasing C parameter (influences the number of violations of the margin allowed) increases bias, but decreases variance
    * The parameterisation of ML algorithms is often a battle to balnce out bias and variance. 

Related: 
    * See :ref:`(Cortes 1995) <cortes1995support>` for SVM. 
    * See :ref:`(Fix 1989) <fix1989discriminatory>` for KNN.
    * See :ref:`(Loh 2011) <loh2011classification>` for DT.
    * See (:ref:`Black 2017 <black2017real>`, :ref:`Black 2019 <black2019rapid>`, :ref:`Boccard 2013 <boccard2013consensus>`) that use LDA.

brudigam2021gaussian
--------------------
Gaussian Process-based Stochastic Model Predictive Control for Overtaking in Autonomous Racing

(Brudigam) uses Gaussain Processes in Reinforcement Learning to design controllers for race cars to overtake. 
    
Related:
    * See :ref:`2022-07-20 - Deep Learning<2022-07-20 - Deep Learning>` where Hayden Dyne discusses this paper. 
    * See :ref:`(Codevilla 2018) <codevilla2018end>`, another racing paper, for RL drifiting controller.

cai2020high 
-----------
    * End-to-end driving via conditional imitation learning. 
    * Model-free reinforcement learning - does not rely on human understanding of world and design controllers. 
    * Human driver is the trajectory with is the goal, uses a professional driver playing the game with a steering wheel. 
    * Model performs on different track difficulties. 
    * Reward function is scaled by velocity, so faster lap times are rewarded. 
    * Works for 4 different kinds of vehicles, although the truck struggles to achieve same performance as lighter ones. 

chase1973perception
-------------------
    * Domain expertise allows people to build meaningful schema to represent patterns.
    * Expert chess players recall 16 pieces, intermeidate 8, novice 4 when arranged in meaninful positions. 
    * Recall was consistant for levels of expertise on nonsense chess boards. 
    * Our mental schemas for encoding patterns break on noise (unseen data). 

chen2019deep
------------
    * Deep reasoning networks: Thinking fast and slow
    * System 1 and System 2 thinking. 

chen2020deep
------------
A deep learning method for bearing fault diagnosis based on cyclic spectral coherence and convolutional neural networks

(Chen 2022) propose a Cyclic Spectral Coherence (CsCoh) + Convolutional Neural Networks (CNNs) for rolling element fault diagnosis. 

Data: 
    * The domain is rolling element fault diagnosis - i.e. ball bearings in a factory setting. 
    * A rotating bearing will modulate (go up and down) in ptich in a non-periodic manner, this is a telltale sign of a faulty ball bearing. 

Method: 
    * Combine CsCoh + CNNs for fault diagnosis of rotating elements in a factory. 
    * Cyclic Speherical Coherence (CsCoh) is used to preprocess virbation signals, estimated by the fourier transform of Cyclic ACF (see paper for derivation). 
    * Group Normalization (GN) is developed to reduce the internal covariant shift by data distribution discrepency, extends applications of the algorithm to real industrial environments. 

Results: 
    * Their proposed method improves classification performance, >95% accuracy needed for use in real-world. 
    * CsCoh proivde superior dsciminate feature representations for bearing health statuses under varying conditions. 
    * Group Normalization increases robustness for data from differenet domains (with different data distributions). 

Why it matters? 
    * Garbage-in-garbage out - Preprocessing can dramatically improve the performance of a CNN.
    * Group Normalization makes the method robust, and applicable to out-of-distribution data from unseen domains. 
    * Detecting faults in ball bearings is crucial for safety, automation, and efficiency in factories.

Related : 
    * See :ref:`2022-10-12 - Deep Learning<2022-10-12 - Deep Learning>` for more. 

chen2019looks
-------------
This looks like that: deep learning for interpretable image recognition

(Chen 2019) forces a deep neural network to use a reasoning process in a human-understandable way. 

Method:     
    * (Chen 2019) forces a deep neural network to use a reasoning process in a human-understandable way. 
    * But while the model's predictions can be explained easily to humans, the parameters of that model remain black-box, an utter mystery.
    * Add a prototype layer to neural networks to for interpretable models for black-box nets. 

chen2021evaluating
------------------
    * 70% accuracy for basic DSA problems. 
    * Can't solve more difficult problems - doesn't optimize solutions for performance. 
    * CoPilot outperforms other state-of-the-art NLP code generation models. 
    * Requires "fine-tuning", supervised human intervention to hint towards correct answer. 

chen2022deep
------------
A deep reinforcement learning framework based on an attention mechanism and disjunctive graph embedding for the job-shop scheduling problem

(Chen 2022) propose Disjunctive Graph Embedded Recurrent Decoding Transformer (DGERD).

Available: https://arxiv.org/abs/1301.3781

Task: 
    * Job shop scheduling:
        * Job shop sechduling refers to the allocation of resrouces, such as machines and operators, subject to certrain constraints. 
        * It  inovles determing order and timing of a set of jobs to be processed.
        * Goal of optimizing one (or more) objective(s), such as minimizing completion time, minimzing delays, or maximizing resource utilization.

Limitations: 
    * Human designed heuristics rely on domain exerptise, and are often sub-optimal. They are static, and cannot adapt to changing conditions.
    * Traditional deep reinforcement learning (DRL) have fixed input size, and fixed parameterization (architecture) that do not generalize well to other problems. 

Method:
    * The job shop scheduling problem can be represneted as a disjunctive graph :ref:`(Balas 1969) <balas1969machine>`.
    * Routing problems can be solved with attention-based representations :ref:`(Kool 2018) <kool2018attention>`.
    * Node2vec :ref:`(Grover 2016) <grover2016node2vec>` is a technique for learning low-dimensional representations of nodes in a graph.
    * Word2vec :ref:`(Mikolov 2013) <mikolov2013efficient>` is a technique for learning low-dimensional representations of words in a corpus.

Related: 
    * Presented at :ref:`2023-10-06 - ECRG <2023-10-06 - ECRG>`
    * node2vec :ref:`(Grover 2016) <grover2016node2vec>`
    * Attention for routing problems :ref:`(Kool 2018) <kool2018attention>`
    * Disjunctive graphs :ref:`(Balas 1969) <balas1969machine>`
    * Attention mechanisms :ref:`(Vaswani 2017) <vaswani2017attention>`
    * Word2vec :ref:`(Mikolov 2013) <mikolov2013efficient>`

Results: 
    * Performs worse than state-of-the-art methods for smaller problems. 
    * Outperforms state-of-the-art methods on on larger problems.
    * Requires re-training for each new problem.
    * GP approaches are competitive with DRL approaches.

chevalier2018babyai
-------------------
    * Babyai: A platform to study the sample efficiency of grounded language learning

codevilla2018end 
----------------
    * High-speed autonomous drifting with deep reinforcement learning. 
    * Far easier to use real-world data on driving that has already been collected than generate simulation data. 
    * Data augmentation used to help network generalize to new scenarios and edge cases not in the training data. 

Related: 
    * See :ref:`(Brudigam 2021) <brudigam2021gaussian>`, another racing paper, for RL overtaking controller. 
    * See :ref:`2022-07-20 - Deep Learning<2022-07-20 - Deep Learning>` where Hayden Dyne discusses this paper. 

cortes1995support
-----------------
    * Cortes and Vapnik proposed the Support Vector Machine (SVM).
    * This model creates a hyperplane that can draw distinct class boundaries between classes.
    * We call these class boundaries the support vectors.
    * We are performing multi-class classification, so it used a one-vs-all approach \cite{sklearn2021feature}.
    * This creates a divide between one class and the rest, then repeats for the other classes.

couillet2022submerged
---------------------
The submerged part of the AI-Ceberg [Perspectives]

(Couillet 2022) provide a critize of AI based on its sustainability and environmental impacts on the planet. 

TODO [ ] Read this paper. 

Related: 
    * See :ref:`2022-11-09 - Deep Learning<2022-11-09 - Deep Learning>`

crall2013hotspotter
-------------------
HotSpotter — Patterned species instance recognition

`(Crall 2013)<https://ieeexplore.ieee.org/abstract/document/6475023>`__ is an instance recognition computer vision paper. 

Purpose: 
    HotSpotter a model to recognize instances based on their unique spots. 
    
Dataset: 
    * This is a species invariant model, that differentiates between dissimilar species, e.g. zebras, giraffes, leopards, and lionfish. Fish and mammals are dissimilar but share spots. 

Method:
    * Local Naive Bayes Nearest Neighbours (:BNN)

Limitations: 
    * relatively dated paper, 2012 paper \cite{mccann2012local} that proposed \acrfull{LNBNN}, 
    * an extension of \acrfull{NBNN} \cite{behmo2010towards}. 
    * where "only the classes represented in the local neighborhood of a descriptor contribute significantly and reliabl to their posterior probability estimates". 
    * The authors admit {LNBNN, did not beat state-of-the-art methods such as feature pyramid networks :ref:`(Lin 2017) <lin2017feature>`, which rely on local soft assignment and max pooling operators. Convolutions and max-pooling are utilized in CNNs \cite{lecun1989backpropagation}, a powerful model for computer vision-related tasks. Which with advancements in hardware, and the lifting of the AI winter, are efficient to train at scale using GPUs. Since then, a a plethor of CNN-based architectures dominate computer-vision tasks:
    
Related: 
     * While images are far from rapid mass spectrometry data, this research aims to perform a similar task, by providing a species-invariant model that differentiates between dissimilar species of fish, e.g. whitefish and oily fish, based on their unique chemical compositions.
     * See :ref:`(Lecun 1989) <lecun1989backpropagation>` for original CNN paper.
     * Local Naive Bayes Nearest Neighrbour (LNBNN) :ref:`(Behmo 2010) <behmo2010towards>`
     * Naive Bayes Nearest Neighbour (NBNN) :ref:`(McCann 2012) <mccann2012local>`

craik1972levels
---------------
    * Levels of processing: A framework for memory research. 
    * Elaborative rehearsal requires deeper processing than maintainence rehearsal. 

craik1975depth
---------------
    * Deeper processing, semantic over structural or phonetic, better. 
    * Depth processing increased later recognition of words in a list. 
    * Annecodte, study: skim-read vs. thoughtful reading. 

da2018evolutionary
------------------
    * Evolutionary Computation Approaches to Web Service Composition. 
    * Service composition is an NP-hard combinatorial problem - local search via heuristic is needed. 
    * Optimizes fitness as multi-objective function of correctness and exectution time. 
    * Graph building algorithm that uses evolutionary techniques, mutation and crossover. 
    * Don't reinvet the wheel, encourage reuse of existing services. 

dawkins1995evolved
-------------------
The Evolved Imagination: Animals as models of their world

(Dawkins 1995) proposed animals are models of their world. 

Available https://richarddawkins.net/1995/09/the-evolved-imagination-animals-as-models-of-their-world-2/ 

Related: 
    * See Wolfgang's talk at 2022-10-28 - ECRG , GP as a model of a discrete fitness landscape. 
    * See 12:18 from "Psychedlics, Consciosness, and AI \| Richard Dawkins \| #256" https://youtu.be/HbGoUwmqIEQ?t=738

devlin2018bert
--------------
Bert: Pre-training of deep bidirectional transformers for language understanding

Available: https://arxiv.org/abs/1810.04805

BERT is a bidrectionanal transformer model proposed by google. 

Related: 
    * :ref:`2023-02-22 - Deep Learning <2023-02-22 - Deep Learning >` discussed here. 
    * :ref:`(Vaswani 2017) <vaswani2017attention>` attention paper

di2019survey
------------
    * A survey on gans for anomaly detection
    * Generative Adversarial Networks (GANs) can be used for anomoly detection. 
    * We build an latent representation of the expected data from nominal samples. 
    * Then measure the reconstruction error between the latent representation and the anomoly.
    * If the reconstruction error is unusually high, then the anomoly is detected.
    * If the reconstruction error is low, then it is likely a nominal sample.
    * Compute the error between the model's original input and output. The sample represents an anomoly if the error exceeds a predefined threshold (Bnomial 2022).
    * Medium article https://medium.com/analytics-vidhya/anomaly-detection-using-generative-adversarial-networks-gan-ca433f2ac287 
    * TODO [ ] - READ     

Related: 
    * :ref:`(Goodfellow 2014) <goodfellow2014generative>` proposed Generative Adversarial Networks (GANs). 
    * See (Goodfellow 2016) Chapter 20, pg. 690, 20.10.4 Generative Adversarial Networks https://www.deeplearningbook.org/contents/generative_models.html

ding2005minimum
---------------
Minimum Redudancy Featyre Selection from MicroArray Gene Expression Data. 

(Ding 2005) is the original Minimum Redundancy - Maximum Relevance (MRMR) paper. 
    
Related: 
    * See :ref:`(Zhao 2019) <zhao2019maximum>` for more recent Uber paper.

do2008expectation
-----------------
What is the expectation maximization algorithm?

(Do 2008) is a nature paper that explains the EM algorithm.

Related:
    * See 2023-02-03 - ECRG where Jiabin uses EM. 

domingos2015master
-----------------
The Master Algorithm: How the Quest for the Ultimate Learning Machine Will Remake Our World

(Domingos 2015) gives a broad introduction for beginners to Artificial Intelligence.

Related: 
    * See :ref:`2023-02-02 - FASLIP<2023-02-02 - FASLIP>`


dong2022survey
--------------
A survey for in-context learning

Available: https://arxiv.org/pdf/2301.00234.pdf

Notes: 
    * Mechanisms for in-context learning (ICL) are unclear 
    * Paradigm for training-free learning. 
    * In-context, learn a new task when presented with a demonstration, without any further optimiztion.
    * Few-shot ICL is possible with a large enough corpus of text and sufficient model complexity.
    * ICL is where a language model can learn a task from a few examples without any further fine-tuning.
    * Tasks are often specified in the text, e.g. a textbook may contain word problems with answers.
    * A task-specific language model can be conditioned to perform a certain task, for example answering word problems.
    * Arbitrary tasks could be learnt by scaling up models and training on a very large corpus - more data and parameters improves task-agnostic performance.
    * While the mechanisms of in context-learning `(Dong 2022) <dong2022survey>` are a mystery, :ref:`(Brown 2020) <brown2020language>` shows that scaling up language models improves task-agnostic few-shot performance.
    * ICL is an "emergent property" of LLMs (airquotes as term is controversial)

Related: 
    * OpenAI GPT-3 
    * :ref:`(Brown 2020) <brown2020language>` LLMs are few shot learners papers 


ecoffet2021first
----------------
First return, then explore

(Ecoffet 2021) propose an RL agent that remembers promising states and returning to such states before intentionally exploring.

Related:
    * See 2022-12-05 - AJCAI #01

eder1995gas
-----------
    * Gas chromatography (GC) is a method that can identify chemicial structures in these fish oils.
    * This produces high-dimensional low sample size data from the fish oils.
    * Chemists compare a given sample to a reference sample to determine what chemicals are present.
    * The existing analytical techniques to perform these tasks are time-consuming and laborious.

eiben2015evolutionary
---------------------
    * From evolutionary computation to the evolution of things - Nature review article.
    * X-band antenneas for NASA Space Technology 5 (ST5) spacecraft 
        * Evolutionary-algorithm based aaporach discovered effective antennea esigns. 
        * Also could adjust designs quckly when requirements changed .
        * One of these antennas was deployed, the first computer evolved hardware in space. 
    * EC has an advantage over manual design.
    * Similar to model-free in reinforcement learning (Cai 2020 - cai2020high, Codevilla 2018 - codevilla2018end)
    * State-of-the-art protein structure prediction 
        * Design an algorithm do develop complex energy functions with genetic programming. 
        * EC great at exploring intractibly large combinatorial search spaces with high evaluation cost. 
    * EC have seperation of concerns, phenotype seperate from fitness, good modularity.
    * EC makes no implicit assumptions about the problem.
    * Trends
        * Automated design and tuning of evolutionary algorithms. 
        * Using surrogate models. 
        * Handiling many objectives 
        * Generative and developmental representations.
    * Crazy futurist ideas for this field, evolutionary factories, artificial bio-silica life, etc... 

eich1975state
-------------
    * State-dependent accessibility of retrieval cues in retneion of categorized list. 
    * Subjects are asked to recall a list of words with and without the influence of marajuana. 
    * Subjects who learn something high, are more likely to retrieve that information high.
    * People can not recall their drug-induced experience easily when they sober up. 

emrah2022imbalance
------------------
An imbalance-aware nuclei segmentation methodology for H&E stained histopathology images
  
(Emrah 2022) proposes a novel nuclei segmentation method for cancer diagnosis in histopathology images.

Available: https://www.sciencedirect.com/science/article/pii/S1746809423001532

Related: 
    * Author discussed in FASLIP at :ref:`2023-11-30 FASLIP <2023-11-30 FASLIP>`
    * Nuclei segmentation dataset :ref:`(Kumar 2019) <kumar2019multi>`'
    * Dice pixel classification layer :ref:`(Shaukat 2022) <shaukat2022state>`

espeholt2022deep
----------------
Deep learning for twelve hour precipitation forecasts

(Espeholt 2022) prepose MetNet-2 that can outperform SOTA for 12 hour precipitation forecasts.

Notes: 
    * TODO read 

eyesenck1980effects
-------------------
    * Effects of processing depth, distinctiveness, and word frequency on retention. 
    * In general distinct stimuli are better remembered than non-distinct ones. 
    * We are more likely to remember things that are out of the blue, or that have a personal connection to us. 

fawzi2022discovering 
--------------------
    * Discovering faster matrix multiplication algorithms with reinforcement learning 
    * Deep Mind - AlphaTensor 
    * Improves Strassman's algorithm for 4x4 matrix multiplication for first time in 50 years.
    * Matrix multiplication is the bedrock of deep learning. 
    * Fast matrix multplication can lead to exponential speedups in deep learning.
    * TODO [ ] - Read this paper 

fahy2009update
--------------
Update of the LIPID MAPS comprehensive classification system for lipids1

Def. lipidomics
    Lipidomics is the study of reaction pathways involved in lipid metabolism within biological systems. The lipidome consists of the lipid profile of a particular sample such as cell, tissue or organism, which can be integrated as a metabolome sub-set

Related: 
    * See Propsoal, lipidomics definition used in glossary.

fix1989discriminatory
---------------------
    * K-nearest neighbours (KNN).

fukushima1982neocognitron
-------------------------
    * Rectified Linear Unit (ReLu) paper. 
    * Activation function for neural networks. 
    * Shares nice properties of linear function. chen2019looks
    * But allows for non-linearities to be captured. 

galanakis2019saving
-------------------
    * Saving Food, 2019, has a chapter on Fish Waste. 
    * 60% of treated fish biomass is discarded as waste. 
    * This can be repuprosed as fish oil (e.g. Omega 3), or fish meal (e.g. animal feed). 
    * Their are a range of other products, such as Geltain, Petpitides, Proteins. 
    * Sustainable fish processing would repurpose the fish waste. 

garnelo2018conditional
----------------------
    * Conditional Neural Processes. 
    * Combine Bayesian optimizationa and Neural Networks. 
    * Use Gaussian Processes (GP) to approximate functions within reasonable confidence. 
    * Neural network, encoder-decoder GAN-like architecture to perform ML tasks. 

gencoglu2019hark
----------------
    * HARK Side of Deep Learning--From Grad Student Descent to Automated Machine Learning
    * Grad Student Descent 
    * **TODO** read this! 

glorot2010understanding
-----------------------
Understanding the difficulty of training deep feedforward neural networks

(Glorot 2010) is the original paper on Xavier initialization.

Available: http://proceedings.mlr.press/v9/glorot10a

Related: 
    * Pytorch: https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1ace282f75916a862c9678343dfd4d5ffe.html


girshick2014rich
----------------
    * Rich feature hierarchies for accurate object detection and semantic segmentation 
    * R-CNNs, Region-based Convolutional Neural Networks.
    * Combine region proposals and CNNs. 
    * See :ref:`2022-10-06 - FASLIP<2022-10-06 - FASLIP>` for more details.

godden1975context
-----------------
    * Context-dependent memory in two natural environments: On land and underwater. 
    * Scuba divers who learn lists of words underwater, best recalled them underwater. 
    * Same true for words learnt on land. 
    * Recall accuracy depends on similarity of context in sensory information. 

gomes2020ensemble
-----------------
On ensemble techniques for data stream regression

(Gomes 2020) talks about ADR-Reg in data stream mining

Related: 
    * See (:ref:`Mouss 2004<mouss2004test>`) for Page-Hinkley method for drift detection.
    * See (:ref:`Bifet 2007<bifet2007learning>`) for ADWIN drift detection algorithm.
    * See :ref:`2023-02-16 - FASLIP<2023-02-16 - FASLIP>` where ADR-Reg is mentioned.

gonick2012cartoon
-----------------
The cartoon guide to calculus

(Gonick 2012) is a great book for learning calculus with heaps of pictures.

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

Related:
    * See 2022-10-31 - Guest Speaker

goodfellow2016deep
------------------
Deep Learning 

(Goodfellow 2016) is a textbook on deep learning.

Available: https://www.deeplearningbook.org/

goodfellow2014generative
------------------------
Generative adversarial networks

(Goodfellow 2014) is the original paper on GANs, a deep learning technique for generating new data, based of a game theoretic approach with discriminator and generator networks.

Related: 
    * See 2022-10-26 Deep Learning 
    * :ref:`(Di 2019) <di2019survey>` for a survey on GANs for anomaly detection.
    * See :ref:`(Goodfellow 2016) <goodfellow2016deep>` Chapter 20, pg. 690, 20.10.4 Generative Adversarial Networks https://www.deeplearningbook.org/contents/generative_models.html

goodman2020weighting
--------------------
Weighting NTBEA for game AI optimisation

Related: 
    * :ref:`(Volz 2018) <volz2018evolving>` same author evolves mario levels using EAs on GAN latent spaces. 
    * :ref:`(Perez 2019) <perez2019analysis>` same author uses RHEA to design Game AI for ponnerman. 

grcic2021densly
---------------
Densely connected normalizing flows

Available:
    * https://proceedings.neurips.cc/paper/2021/hash/c950cde9b3f83f41721788e3315a14a3-Abstract.html

Notes: 
    * Normalizing flows are bijective mappings between input and latent representations with a fully factoritzed distribution. 
    * Normalizing flows (NF) are attrictive due to exact likelihood evaluation and efficient sampling. 
    * However their effective capacity is often insuffiencet since bijectivity constraints limit the model width. 
    * The proposed method addresses this limitation by incrementally padding intermediate representations with noise. Precondition noise in accordance with previous invertible units, coined "cross-unit coupling".
    * Their invertible glow0like, modules increase the expressivity by fusing a densely connected block with NYstron self-attention. 
    * They refer to their proposed achitecture as DenseFlwo, since both cross-unit and intra-module couplings rely on dense connectivity. 
    * Experiments show significant improvements due to prposed contributions and reveal state-of-the-art density estimation under moderate computing budgets. 

grover2016node2vec
------------------
node2vec: Scalable Feature Learning for Networks

(Grover 2016) is a paper on node2vec, a method for learning low-dimensional representations of nodes in a graph.

Available: https://dl.acm.org/doi/abs/10.1145/2939672.2939754

Related: 
    * Mentioned in :ref:`2023-10-06 - ECRG <2023-10-06 - ECRG>`

handa2006robust
---------------
Robust route optimization for gritting/salting trucks: A CERCIA experience

(Hand 2006) use evolutionary computation for route optimization for gritting trucks. 

Related: 
    * :ref:`(Li 2002) <li2002novel>` use evolutionary computation to solve differentiral equations for deriving physics laws. 
    * :ref:`(Li 2002) <li2002novel>` is another paper by same author, with EC for solving DE in materials science.
    * :ref:`(Runarsson 2000) <runarsson2000stochastic>` used stocastic ranking (bubblesort variant) for constrained optimization with Evolutionary Computaiton.

hand2001idiot
-------------
Idiot's Bayes—Not So Stupid After All?

(Hand 2001) is a paper that discusses the Naive Bayes classifier.

Available: https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1751-5823.2001.tb00465.x

Notes:
    * Despite the assumption of independence, Naive Bayes is a powerful classifier.
    * Naive bayes assumption is that the features are conditionally independent given the class.
    * This assumption is not always true, but the model still performs well in practice.

haralick1973textual
-------------------
Textural Features for Image Classification

(Haralick 1973) propose grey-level co-occurence matrix for image analysis.

Available: https://ieeexplore.ieee.org/abstract/document/4309314

Related: 
    * Discussed in :ref:`2023-08-10 - FASLIP <2023-08-10 - FASLIP>`
    * Sklearn documentation and code available: https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_glcm.html
    * Naive bayes. 

he2016deep
----------
Deep residual learning for image recognition

(He 2016) is the original paper on ResNet.

Available: http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

Notes: 
    * A residual neural network (He 2016) is a deep learning model in which the weight layers learn redidual functions with reference to the layer inputs. 
    * Defn. a network with skip connectionts that perform identity mappings, merged with layer outputs by addition. 
    * (He 2016) proposed ResNet for imace Recognition, the original Resnet paper that won the ILSVRC 2015 classification task. Residual neurons, or skip connetions between layers.
    * Skip connections provide shortcuts for information flow between layers of a nerual network. Skip connections allow a network to better propogage information between layers, which inproves performance overall. * A residual neural network (He 2016) is a deep learning model in which the weight layers learn redidual functions with reference to the layer inputs. 
    * Defn. a network with skip connectionts that perform identity mappings, merged with layer outputs by addition. 
    * (He 2016) proposed ResNet for imace Recognition, the original Resnet paper that won the ILSVRC 2015 classification task. Residual neurons, or skip connetions between layers.
    * Skip connections provide shortcuts for information flow between layers of a nerual network. Skip connections allow a network to better propogage information between layers, which inproves performance overall. 

Related: 
    * Dicussed in :ref:`2023-05-25 - FASLIP <2023-05-25 - FASLIP>`
    * See :ref:`(Lecun 1989) <lecun1989backpropagation>` for LeNet.
    * See :ref:`(Krizhevsky 2012) <krizhevsky2012imagenet>` for AlexNet.
    * See :ref:`(Simonyan 2014) <simonyan2014very>` for VGGNet.
    * See :ref:`(Szegedy 2015) <szegedy2015going>` for GoogLeNet.
    
he2020bayesian
--------------
Bayesian Deep Ensembles via the Neural Tangent Kernel

TODO: 
    * read https://proceedings.neurips.cc/paper/2020/hash/0b1ec366924b26fc98fa7b71a9c249cf-Abstract.html


hengzhe2021evolutionary
-----------------------
An Evolutionary Forest for regression

(Hengzhe 2021) is a TVEC paper for Evolutionary Forest.

Related:
    * See :ref:`2023-02-02 - FASLIP<2023-02-02 - FASLIP>`

hendrycks2016gaussian
---------------------
Gaussian error linear units (gelus)

(Hendrycks 2016) is the original paper on Gaussian error linear units (GELUs).

Available: https://arxiv.org/abs/1606.08415

:math:`GELU(x) = 0.5 * x * (1 + Tanh(\sqrt{2/\pi} * (x + 0.044715 * x^3)))`

Related: 
    * pytorch https://pytorch.org/docs/stable/generated/torch.nn.GELU.html

hildebrandt2010towards
----------------------
Towards improved dispatching rules for complex shop floor scenarios: a genetic programming approach

(Hildebrandt 2010) use genetic programming for dispatching rules in complex shop floor scenarios.

Available: https://dl.acm.org/doi/abs/10.1145/1830483.1830530

hinton2012improving
-------------------
Improving neural networks by preventing co-adaptation of feature detector

Available https://arxiv.org/abs/1207.0580

Notes:
    "This "overfitting" is greatly reduced by randomly omitting half of the feature detectors on each training case" - abstract

Related:
    * Dropout paper :ref:`(Srivastava 2014) <srivastava2014dropout>`

ho1995random
-------------
Random decision forests

(Ho 1995) is the original paper on random forests.

Available: https://ieeexplore.ieee.org/abstract/document/598994/

Notes: 
    * Random forest.

ho2020denoising
---------------
Denoising diffusion probabilistic models

Related: 
    * :ref:`(Song 2020)<song2020denoising>` proposed DDIM, a generalized DDPM that is faster.
    * Stable Diffusion https://github.com/CompVis/stable-diffusion
    * Deforum Notebook https://t.co/mWNkzWtPsK
    * See :ref:`2023-05-03 - Deep Learning <2023-05-03 - Deep Learning>`

hof2016industrial
-----------------
The industrial melanism mutation in British peppered moths is a transposable element

(Hof 2016) moth that changes colour of its wings due to transposons. 

* TODO [ ] Read this paper.
* Nature article 

Related: 
    * Julie ECRG - 2022-10-14 mentioned this. 
    * :ref:`(Bourque 2018) <bourque2018ten>` explains transposons in detail.
    * :ref:`(Kulasekara 2014) <kulasekara2014transposon>` says changes passed to offspring. 

Hofstadter1979godel 
-------------------
    * Godel Escher Bach 
    * The hand that draws itself. 

howard2017mobilenets
--------------------
Mobilenets: Efficient convolutional neural networks for mobile vision applications

Available: https://arxiv.org/abs/1704.04861

Related: 
    * Discussed in :ref:`2023-09-21 - FASLIP <2023-09-21 - FASLIP>`


huang2017densely
----------------
Densely connected convolutional networks

(Huang 2017) is the original paper on DenseNet, a deep learning technique for image classification.

Available: https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html

Related:
    * Discussed in :ref:`2023-09-21 - FASLIP <2023-09-21 - FASLIP>`

hung2019optimizing
------------------
Optimizing agent behavior over long time scales by transporting value

(Hung 2019) deal with naviagation with distraction, a model that requires semantic control.

Related:   
    * See 2022-12-05 - AJCAI #01

hussain2016food
---------------
Food contamination: major challenges of the future

Def. Food contamination: 
    Food contamination is generally defined as foods that are spoiled or tainted because they either contain microorganisms, such as bacteria or parasites, or toxic substances that make them unfit for consumption. A food contaminant can be biological, chemical or physical in nature, with the former being more common. These contaminants have several routes throughout the supply chain (farm to fork) to enter and make a food product unfit for consumption.

Related: 
    * See proposal, fish contamination deteciton. 

huszar2022algorithmic
---------------------
Algorithmic amplification of politics on Twitter

(Huszar 2022), study by former Twitter employees, reveal amplification of political content on Twitter.

Related:
    * Discussed in Deep Learning - 2022-11-30

ioffe2015batch
--------------
Batch normalization: Accelerating deep network training by reducing internal covariate shift

Available: https://arxiv.org/abs/1502.03167

:math:`y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta`

Notes: 
    * Batch Normalization is a popular technique used to train deep neural networks. It normalizes the input to a layer during every training iteration using a mini-batch of data. It smooths and simplifies the optimization function leading to a more stable and faster training.
    * Batch Normalization works by scaling its input—the previous layer's output—to a mean of zero and a standard deviation of one per mini-batch.
    * Although correctly initializing a network can significantly impact convergence, the stability offered by Batch Normalization makes training deep neural networks less sensitive to a specific weight initialization scheme. Since Batch Normalization normalizes values, it reduces the likelihood of running into vanishing or exploding gradients.
    * Batch Normalization does require extra computations, making individual iterations slower. However, it will dramatically reduce the number of iterations needed to achieve convergence, making the training process much faster.
    * However, at initialization, batch normalization in fact induces severe gradient explosion in deep networks. Practically, this means deep batchnorm networks are untrainable.
    * This is only relieved by skip connections in the fashion of residual networks :ref:`(He 2016) <he2016deep>`

Related: 
    * :ref:`(He 2016) <he2016deep>` ResNet fixes gradient explosion in deep networks with batchnorm. 
    * :ref:`(Szegedy 2015) <szegedy2015going>` GoogLeNet - same author. 
    * :ref:`(Szegedy 2013) <szegedy2013intriguing>` same author.
    * Pytorch 1D https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

ingalalli2014multi
------------------
A multi-dimensional genetic programming approach for multi-class classification problems

(Ingalalli 2014) propose M2GP, for feature construction for mutli-class classification tasks.

Available: https://link.springer.com/chapter/10.1007/978-3-662-44303-3_5

Notes: 
    * M2GP is a multi-dimensional genetic programming approach for multi-class classification problems.
    * Fixed number of dimensions :math:`d`
    * Predecessor to M3GP 2023-10-06 - ECRG

Related: 
    * Discussed in FASLIP :ref:`2023-09-28 - FASLIP <2023-09-28 - FASLIP>`. 

jacot2018neural
---------------
    * Neural tangent kernel: Convergence and generalization in neural networks

jaegle2021perceiver
-------------------
Perceiver: General perception with iterative attention

(Jaegle 2021) is a DeepMind paper on a multi-modal perceptron with attention.

Related:
    * See :ref:`2023-03-01 - Deep Learning<2023-03-01 - Deep Learning>` for discussion on this paper.

jha2015rapid
------------
Rapid detection of food adulterants and contaminants: theory and practice

Def. adulteration: 
    Food adulteration is the act of intentionally debasing the quality of food offered for sale either by the admixture or substitution of inferior substances or by the removal of some valuable ingredient 

Related:    
    * :ref:`(Black 2019)` uses REIMS to detect beef adulteration. 

jiang2019degenerate
-------------------
Degenerate Feedback Loops in Recommender Systems

(Jiang 2019) is a deep mind paper on degeneracy in positive feedback loops on social media.

Related:
    * See Deep Learning - 2022-11-30 for discussion on this paper.

jing2020learning
----------------
    * Graph nerual Networks can be used for protien folding. 
    * Equivariance to rotations - if the networks thinks the same instance rotates is a completely different structure, this is very inefficient. 
    * Instead we want rotation invariant representations for things like protiens. (Like we wan't time invariant representations for gas chromatography). 
    * Voxels are 3D pixels, these can be used to make a 3D representation of an instance, which then applies a 3D Convolutional Neural Network. 
    * We think that (1) message passing and (2) spatial convolution, are both well suited for different types of reasoning. 
    * In protein folding, their are chemical propoerties of protiens that simplify the combinatorial search space for the graphical neural network. 
    * This is similar to how the AI Feynman (Tegmark 2020) used properties of physics equations to simplify symbolic regression. 

jing2022masked
--------------
Masked siamese convnets

Task: 
    * low-shot image classification and outperforms previous methods on object detection benchmarks

Data: 
    * object detection benchmarks
    
Related: 
    * :ref:`(Bromley 1993) <bromley1993signature>` is the original siamese network paper.
    * :ref:`(Zhu 2020) <zhu2020masked>` propose siamese networks for ransomware detection.

kajiya1993get
-------------
    * How to get your SIGGRAPH paper rejected
    * TODO [ ] Read this

karras2020analyzing
-------------------
    * StyleGAN 
    * Latent layer representation. 
    * Manipulating latent layer gives a sense of semantically meaninful feature space. 
    * We can see the change in style that sampling latent layer gives. 

Related: 
    * See :ref:`(Karras 2022) <karras2022elucidating>` for LDM design space paper from same author.

karras2022elucidating
---------------------
Elucidating the design space of diffusion-based generative models

(Karras 2022) provides a clear explanation of the design of generative models.

Background:
    * Diffusion-based generative models were unnecessarily convoluted. 

Motivation:
    * Simplify Latent Diffusion Model (LDM) architecture, decouple architecture,
    * Provide a clear explanation of the design space of generative models.

Data: 

Method:

Results:

Why it matters? 
    * NeurIPS 2022 paper for LDMs, provided code that EVERYBODY uses (steals!)

Limitations: 
    * No video, consistency across time, recurrence needed. 

Applications: 
    * (Wood 2022) Glimpse of Us - Joji (AI Generated Music Video) https://youtu.be/IzhWOuCzzzs
    * Deforum Art - Twitter profile https://twitter.com/deforum_art

Related: 
    * See :ref:`(Karras 2020) <karras2020analyzing>` for StyleGAN paper from same author.
    * See :ref:`2023-05-03 - Deep Learning <2023-05-03 - Deep Learning>`

karpathy2023lets
----------------
Let's build GPT: from scratch, in code, spelled out.

(Karpathy 2023) builds GPT from scratch 

YouTube https://youtu.be/kCc8FmEb1nY?si=1vM4DhyqsGKUSAdV

Related:
    * Transformer :ref:`(Vaswani 2017) <vaswani2017attention>`
            
katharopoulos2020transformers
-----------------------------
Transformers are rnns: Fast autoregressive transformers with linear attention

`(katharopoulos 2020) <https://proceedings.mlr.press/v119/katharopoulos20a.html>`__ propose :math:`O(n)` transformers with self-attention as a linear dot-product of kernel feature maps.

Notes: 
    * Transformers achieve remarkable performance in several tasks but due to their quadratic complexity, with respect to the input's length, they are prohibitively slow for very long sequences. 
    * To address this limitation, we express the self-attention as a linear dot-product of kernel feature maps and make use of the associativity property of matrix products to reduce the complexity from :math:`O(n^2)` to :math:`O(n)`, where N is the sequence length. 
    * We show that this formulation permits an iterative implementation that dramatically accelerates autoregressive transformers and reveals their relationship to recurrent neural networks. 
    * Linear Transformers achieve similar performance to vanilla Transformers and they are up to 4000x faster on autoregressive prediction of very long sequences.

Related: 
    * Discussed in :ref:`2023-05-10 - Deep Learning <2023-05-10 - Deep Learning>`
    * See :ref:`(Zhai 2021) <zhai2021attention>` for Attention Free Transformer (AFT)
    * See :ref:`(Peng 2023) <peng2023rwkv>` for RWKV - transformers + RNNs. 
    * See :ref:`(Wang 2020) <wang2020linformer>` for Linformer paper.
    * See :ref:`(Kitaev 2020) <kitaev2020reformer>` for Reformer paper. 
    * See :ref:`(Vaswani 2017) <vaswani2017attention>` for transformer paper. 

ke2018sparse
------------
    * Sparse attentive backtracking: Temporal credit assignment through reminding

kennedy1995particle
-------------------
Particle Swarm Optimisation (PSO). 

Purpose: 
    * PSO optimizes non-linear functions with particle swarn methedology. 
    * PSO was discovered through simulation of a simpleified social behaviour model. Then taken from a social behaviour model, and turned into an optimizer. 

Background: 
    * The synchonicit was though of as a function of the bird trying to maintain an optimal distance between itself and its neighbours.
    * All birds in the flock know the global best position, the roost. 
    * (Millonas 1995) developed 5 basic principles of swarm intelligence. 
        1. Prxomity - perform space/time computations. 
        2. Quality - respond to quality features in the environment 
        3. Diversity - not commit to narrow channels. 
        4. Stablity - Don't change mode behaviour each iteration. 
        5. Adaptability - Change behaviour if it is worth it. 
    * Paradigms: 
        1. Artificial life - i.e. fish schooling, birds flocking, 
        2. Genetic algorithms / evotionary programming. 
    * Train ANN weights, Model Schaffers f6 function a GA from (Davis 1991).
    * School of Fish https://youtu.be/15B8qN9dre4
    * (Heppner 1990) had simulations which introduced a "roost", a global maximum, or home the birds, that they all know. 
    * But, how do birds find food? I.e. a new bird feeder is found within hours. 
    * Agents move towards their best know value - the cornfield, in search of food.
    * Birds store their local maxima, the cornfield vector (I know there is food here!).  
    * Model is very simple, requires a few lines of code, primitive mathematics operators, both effecient in memory and speed. 
    * (Reynolds 1987) was intrigued by the aesthetics of bird flocking, the choreography, synchonocity. He wanted to understand the mechanics of bird flocking - as set of simple rules that governed the behaviour. 
    * With the assumption, like Conway's Game of Life for cellular automata, that a simple set of rules, my underpin the unpredictable and complex group dynamics of bird social behaviour. 

Motivations: 
    * Motivation for simulation: to model human behaviour. Humans are more complex, we don't just update our velocity/direction as animals flocking do, we update our beliefs/views to conform to our peers around us - i.e. social desirability bias, cultural homogenuity. 

.. Data: 

Method: 
    * Explorers and settlers model, explorers overrun target, settlers more precise, had little improvement, Occam's razor removed the complex model. 
    * Initial approach: a nearest neighbour method to synchonocity that matched velocity resulted in unifrom unchanging direction. 
    * Stochasity, randomness, "craziness" was required to add variation to the flocks direciton. Enough stochacity to give the illusion of aritificial life. 
    * Simulation behaviour: a high p/g increment had violent fast behaviour, an approximately equal p/g increment had synchronocity, low p/g increment had no convergence.
    * Improvements: removed craziness, removed nearest neighbour (NN), without NN collisions were enabled, the flock was now a swarm. A swarm not a flock, because we have collisions. 
    * g/p increment values had to be chosen carefully. 
    * Social anaologies: :math:`pbest` is autiobiographical memory, :math:`\nabla pbest` is simple nostalgia. :math:`gbest` is public knowledge, :math:`\nabla gbest` is social conformity. 
    * Appxomiations, PSO could solve the XOR problem on a 2-3-1 ANN with 13 parameters. 
    * Improvement: velocities were adjusted according to their difference, per dimension, this added momementum, a memory of previous motion. p/g increment was a nuisance parameter, and was such removed. 
    * Stochastic factor, which amplifieid the randomness, was set to 2. This makes the agents "overfly" or overshoot the target about half of the time. Tuned with black magic, a more formal derivation could be done in future work. 
    * Tried a model with one midpoint between :math:`gbest` and pbest, but it converged at the midpoint. 
    * The stochasity was necesarry for good results. 
    * Version without momentum, had no knowledge of previous motion, and failed to find the global optima. 
    
Results: 
    * PSO met all 5 of (Millonas 1995) swarm intelligence principles: 
        1. n-d space calucaltions computed over a series of time setps. 
        2. Responds to quality factors :math:`gbest` and pbest. 
        3. Moves between :math:`gbest` and pbest, encourging diversity. 
        4. Mode behaviour only changes when :math:`gbest` does. 
        5. Mode behaviour does change when :math:`gbest` does. 
    * Term particle chosen as birds have velocity and acceleration, similar to elementary particles in phusocs. (Reeves 1983) also dicussed particle systems and primitive particles as models of diffucse objects, like a cloud of smoke. So we can refer to the representation as a particle swarm. 
    * PSO sometimes find ANN weights better than those found via gradient descent. 
    * PSO is a form of Evolutionary Computation, somewhere between genetic algorithms and evolutionary programming.
    * :math:`gbest` / :math:`pbest` is similar to crossover operator, it also has a fitness function, both from evolutionary computation (EC).
    * The momentum of the swarm flying towards better solutions, and often overshooting, is a strength. IT allows the swarm to explore unkown regions in the problem domain. 

Applications: 
    1. non-linear function optimization, 
    2. neural network training. 

Philosophy (some beautiful philosophical musings from the end of the paper): 
    * Perhaps these same rules govern social behaviour in humans. Social sharing of infomration amoung members of the same species (cospeciates) offers an evolutionary advantage (Wilson 1975).
    * In abstract multi-dimenisional space, our psychological space, we allow colluions within a population - i.e. two individuals may share the same beliefs. Thus our model allows collisions, e.g. "collision-proof birds". 
    * Aristotle spoke of Qualitative and quantitative movement. 
    * PSO walks a fine line between order (known) and chaos (unknown). 
    * Allows wisom to emerge rather than impose it. 
    * Emulates nature rather than trying to control it. 
    * Makes things simpler than more complex.

Related: 
    * :ref:`(Kennedy 1997) <kennedy1997discrete>` Discrete PSO, for feature selection.
    * :ref:`(Wood 2022) <wood2022automated>` uses PSO for feature selection in GC-MS data.

kennedy1997discrete
-------------------
PSO for feature selection. 

Notes: 
    * TODO [ ] Read this paper.

Related: 
    * :ref:`(Kennedy 1995) <kennedy1995particle>` original PSO paper. 

kerber1992chimerge
------------------
Chimerge: Discretization of numeric attributes 
   
Notes: 
    * Predecessor to Chi2 (Liu 1995, liu1995chi2)

Related: 
    * :ref:`(Liu 1995) <liu1995chi2>` the successor to Chimerge. 
    
khakimov2015trends
------------------
Trends in the application of chemometrics to foodomics studies

Notes: 
    * TODO [ ] READ THIS !!! 

Daniel email:
   * Re: using the 4800x500 image, would it be possible to use a three dimensional ‘data cube’ instead of a 2D image? i.e. time x peak intensity x mass spectrometry (See image below I took from the attached paper)? When we started the work on the GC data, that was the kind of format I hoped to use.
   
Why it matter? 
   * Data cube, a useful representation of GS-MS data. 
   
Related: 
   * :ref:`(Bi 2022) <bi2020gc>` proposed a CNN model that incorporated GC-MS data fusion for food science.
   * :ref:`(Zhang 2008) <zhang2008two>` proposed a 2-D COW algorithm for aligning gas chromatography and mass spectrometry.
   * :ref:`(Eder 1995) <eder1995gas>` The original paper on gas chromatrography (GC). 
  
killeen2017fast
---------------
Fast sampling, analyses and chemometrics for plant breeding: bitter acids, xanthohumol and terpenes in lupulin glands of hops (Humulus lupulus)

(Killeen 2017) addressed rapid chemical analysis techniques for hops. 

Related: 
    * See 2023-02-08 - Callaghan Innovation Workshop, for Daniels talk on this paper. 

kingma2014adam
--------------
    * Adam optimizer for neural networks. 

kira1992practical
-----------------
    * A practical approach to feature selection,
    * Relief feature selection method, predecessor to ReliefF (Kononeko 1994, kononenko1994estimating)
    * Authors suggest: splitting into a sereis of 2-class problems to handle multi-class problems. 

kishore2021fixed
----------------
    * Hide messages in adversarial neural network. 
    * Pre-trained stenograph, results in non-zero error, we need perfect reconstruction for encryption.
    * Face anonymization, post a persons face online, then regenerate the face, but encrypt the private face. 
    * This lets friends anonmyously share images with their face online, without revealing their identity.

kitaev2020reformer
------------------
Reformer: The efficient transformer

`(Kitaev 2020) <https://arxiv.org/abs/2001.04451>`__ propose the Reformer, an :math:`O(L\log L)` efficient transformer.

Notes: 
    * replace dot-product attention by one that uses locality-sensitive hashing, changing its complexity from :math:`O(L^2)` to :math:`O(L\log L)`, where :math:`L` is the length of the sequence

Related: 
    * Discussed in :ref:`2023-05-10 - Deep Learning <2023-05-10 - Deep Learning>`
    * See :ref:`(Wang 2020) <wang2020linformer>` for linformer. 
    * See :ref:`(Peng 2023) <peng2023rwkv>` for RWKV transformer + RNNs paper. 
    * See :ref:`(Zhai 2021) <zhai2021attention>` for attention free transformer (AFT paper).
    * See :ref:`(Wang 2020) <wang2020linformer>` for Linformer paper.
    * See :ref:`(Katharopoulos 2020) <katharopoulos2020transformers>` for linear transformers. 
    * See :ref:`(Vaswani 2017) <vaswani2017attention>` for transformer paper. 

kobyzev2020normalizing
----------------------
Normalizing flows: An introduction and review of current methods

Related:    
    * See :ref:`2022-10-26 - Deep Learning<2022-10-26 - Deep Learning>`

kobyzev2020normalizing
----------------------
Attention, Learn to Solve Routing Problems!

(Kobyzev 2020) propose a transformer for solving routing problems.

Related: 
    * See :ref:`2023-10-06 - ECRG <2023-10-06 - ECRG>`
    * See :ref:`(Grover 2016) <grover2016node2vec>` for node2vec.
    * See :ref:`(Balas 1969) <balas1969machine>` for disjunctive graph.

kononenko1994estimating
-----------------------
Estimating attributes: Analysis and extensions of Relief. 
    

Notes: 
    * ReliefF paper
    * ReliefF feature selection method. 
    * Original Relief method (Kira 1992), could not handle multi-class problems. 
    * Contributions: extend Relief (Kira 1992) to ReliefF (Kononeko 1994) to handle 
        * noisy, 
        * missing features, and, 
        * multiclass problems. 
    * Motivation: Heuristics needed to identify features woth strong depednenceies due to combinatorial explosion in high-dimensional data. 
    * Information gain and mutual information are equivalent, MI is used for MRMR. 
    * Key idea: estimate atttributes according to how well their values distinguish amoung instances that are near eachother. 
    * Relief Searches for 2 closest neighbours, one of same class (hit), one of different (miss). Then compares attributes ability to seperate the hit and miss. 
    * Rationale: a goof attribute can differentiate instances from different classes. And should have the same value for nearest neighbour of the same class. 
    * Extensions to handle: noise, incomplete data, and multi-class problems.
    * Diff calculates distance from :math:`V` to the hit and miss. 
    * The algorithm is an approximation of the distance metric: :math:`W[A]=P(different value of A | miss) - P(different value of A | hit)`. 
    * Limitations of Relief (Kira 1992): 
        * Noisy/redundant features will strongly affect selection of nearest neighbours. 
        * Estimiation of attributes :math:`W[A]` becomes unreliable on noise data. 
    * Fix: Take K nearest neighbours for hit/miss, to increase the reliability og probablity apporximiation, and average (A) the result, hence Relief-A. 
    * :math:`m` is a normalization constant, :math:`m` caanot exceed the number of training instances, :math:`m \ge |T|`, where :math:`T` is the training set, and :math:`|T|` is its size. 
    * :math:`m` is derieved iteratiely, with :math:`m=|T|` as an upper bound. Similar to how the first phase of chi2 (Liu 1995) determines a good :math:`\chi^2` threshold. 
    * Synthetic dataset with noisy features, these have no/noisy relation to the class variable. Three datasets of increasing order complexity of dependent relationships. 
    * First dataset: 5 noise variables, 5 independent/informative, both in decreasing :math:`P(.)` so some are more important than others. 
    * Second dataset: XOR operator, introduces parity relation of the second order. It introduces a non-linearity, it will have zero covariance, but are not independent. Instead, one attribute that determines the redundancy of two others. 
    * Third dataset: a parity relationship of the third order. 
    * Information gain / mutual info is not equivalent to intended information gain. 
    * Increasing the number of nearest neighbours :math:`n` has a drastic effect on handling noise in the dataset. 
    * Monothously, enitirely non-decreasing or non-increasing. "Line goes up!". 
    * Relief-A performs well on first two datasets, poorly on third. 
    * As :math:`n` increased, the estimaotr of attributes becomes vanishingly similar to the gini index. See (Kononeko 1994) for derivation/proof. 
    * Gini index is an impurity function that is highly corelated with infomration gain/mutual info. 
    * Relief A, as :math:`n` increases approaches high correlation with gini index and mutual info. 
    * There is a limit for :math:`n` neighbours, accuracy collapses when :math:`n` can no longer capture clusters of the same class in the distribution space. 
    * Noise has a drastic effect on data with fully independnet vvariables. Less so for depedend attributes from second/third datasets - perhaps because their are less incorrecly labelled instances in those. 
    * Relief-A,B,C etend Relief in different ways to deal with incomplete datasets. All done through changing the diff function. 
    * Relief-C ignores missing values, and normalizes afterwards - with enough data, it should converge to the right estimate. 
    * Conditional probabilities are approximated using relative frequency in the training set. 
    * Relief-A,B,C had little accuracy difference for datasets without missing values. 
    * Relief-D performed best for all datasets with missing values. 
    * Relief-D calculates the probablity that two given instances have a different value for a given attribute. 
    * Authors (Kira 1992) suggest: splitting into a sereis of 2-class problems to handle multi-class problems. 
    * Relief-E,F extend Relief-D to deal with multi-class problems. 
    * Relief-E, nearest miss becomes nearest neighbour for a different classes. A simple and straightforward extension. 
    * Relief-F, takes weighted average of near miss from each class, rather than just one class, as in Relief-E. 
    * Algorithm can seperate each pair of classes regardless of which two classes were closest. Robust to all classes becayse of weighted average. 
    * Relief-F outperforms Relief-E for all synthetic datasets. Both with/without noise. 
    * Most important contribution: allow Relief-F to deal with multi-class problems. 
    * Tumour dataset is a real-world dataset with independent variables (verified by domain experts - phycisians). 
    * :math:`W[A]` is an approxmiation of the information gain of attributes, higher correlation means this approximiationj is closer to the true mutual information. 
    * Issues with Relief-F: it can not handle multi-valued attributes. 
    * Other methods overestimate with mutual infomraiton according to domain experts. 
    * Relief-F and normalized mutual infomration estimates important features for the tumour dataset correctly.
    * Myopy - narrow-minded/focussed on a single idea.  
    * Calls out reviewer in the acknowledgements section. 

Related: 
    * Mutual information can be given for a discrete and continuos by a double sum and integral respectively. See :ref:`(Goodfellow 2016) <goodfellow2016deep>` chapter 3 pg. 72 for a derivation of Kullback-Leibler divergence. 
    * :ref:`(Kira 1992) <kira1992practical>` an extension of Relief
    * :ref:`(Wood 2022) <wood2022automated>` used Relief-F for feature selection benchmark. 

kool2018attention
-----------------
Attention, learn to solve routing problems!

(Kool 2018) propose a transformer for solving routing problems.

Available: https://arxiv.org/abs/1803.08475

Related: 
    * Discused in :ref:`2023-10-06 - ECRG <2023-10-06 - ECRG>`
    * Transformers for job shop scheduling :ref:`(Chen 2022) <chen2022deep>`
    * node2vec :ref:`(Grover 2016) <grover2016node2vec>`
    * Disjunctive graphs :ref:`(Balas 1969) <balas1969machine>`
    * Attention mechanisms :ref:`(Vaswani 2017) <vaswani2017attention>`
    * Word2vec :ref:`(Mikolov 2013) <mikolov2013efficient>`

koppen2000curse
---------------
Curse of dimensionality. 

(Koppen 2000) discussed the curse of dimensionality.

Available: https://www.class-specific.com/csf/papers/hidim.pdf

Related:
    * Discussed by Ruwang in :ref:`2023-12-08 - ECRG <2023-12-08 - ECRG>`

kulasekara2014transposon
------------------------
Transposon mutagenesis

Notes: 
    * Transposons effects are passed on to offsrping, because their effects are encorporated into the genome. 

Related: 
    * :ref:`(Hof 2016) <hof2016industrial>` discussed tranposons affect on Moths. 
    * :ref:`(Bourque 2018) <bourque2018ten>` discussed transposons in general. 
    * Julie discussed this in 2022-10-14 - ECRG 

kumar2019multi
--------------
A multi-organ nucleus segmentation challenge

Available: https://ieeexplore.ieee.org/abstract/document/8880654

Related:
    * Discussed in :ref:`2023-11-30 FASLIP <2023-11-30 FASLIP>`
    * Mentioned in :ref:`Emrah 2022 <emrah2022imbalance>`

krizhevsky2012imagenet
----------------------
Imagenet classification with deep convolutional neural networks

(Krizhevsky 2012) proposed AlexNet.

Related: 
    * :ref:`(Krizhevsky 2017) <krizhevsky2017imagenet>` further AlexNet paper.
    * :ref:`(Lecun 1989) <lecun1989backpropagation>` proposed LeNet, the original CNN.

krizhevsky2017imagenet
----------------------
Imagenet classification with deep convolutional neural networks

(Krizhevsky 2012) improved AlexNet

Related: 
    * :ref:`(Lecun 1989) <lecun1989backpropagation>` proposed the original form of LeNet

kullback1951information
-----------------------
On information and sufficiency

Notes: 
    * Kullback-Leibler (KL) divergence. 
    * Measures distance between two probability distributions. 
    * Most common loss function for deep learning with stochastic gradient descent. 

Related: 
    * :ref:`(Goodfellow 2016) <goodfellow2016deep>` chapter 3 pg. 72 for a derivation of Kullback-Leibler divergence.


lecun1989backpropagation
------------------------
Backpropagation applied to handwritten zip code recognition

(Lecun 1989) proposed the original form of LeNet

Motivations: 
    * CNNs are a special case of multilayer perceptrons (MLPs).
    * MLPs are not translation invariant.
    * MLPs are not robust to distortions in the input.

Dataset: 
    * MNIST handwritten digits dataset.
    * 60,000 training images, 10,000 test images.

Method: 
    * Architecture is called the LeNet-5.
    * Model consists of: Convolutional layers, Pooling layers, MLP layers.
    * Convolution and pooling layers perform automatic feature extraction.
    * Fully connected layers learn to perform classification based on the extracted features.
    * LeNet-5 Architrecture: 
        1. Input layer: The input layer takes in the 28x28 pixel grayscale images of handwritten digits from the MNIST dataset.
        2. Convolutional layers: The first convolutional layer applies six filters to the input image, each filter being 5x5 pixels in size. The second convolutional layer applies 16 filters to the output of the first layer.
        3. Subsampling layers: The subsampling layers perform down-sampling on the output of the convolutional layers, reducing the dimensions of the output. The subsampling is done using a max-pooling operation with a 2x2 window.
        4. Fully connected layers: The output of the subsampling layers is then passed through three fully connected layers, with 120, 84, and 10 neurons, respectively. The final layer has 10 neurons, each representing a possible digit from 0 to 9.

Results:
    * 99.2% accuracy on MNIST test set.
    * 0.8% error rate on MNIST test set.

Why it matters? 
    * CNNs are a powerful architecture for computer vision tasks. 
    * CNNs recognique local connectivity in data that is spatially related (e.g. images).
    * CNNs are translation invariant.

Limitations: 
    * CNNs are not rotation invariant.
    * CNNs are not scale invariant.
    * CNNs are not robust to distortions in the input.

Related: 
    * :ref:`(Lecun 1998) <lecun1998gradient>` describres practical applications for CNNs.
    * :ref:`(Lecun 1989) <lecun1989generalization>` describes the generalization ability of CNNs.
    * :ref:`(Lecun 1989) <lecun1989handwritten>` describes practical applications of CNNs for handwritten digit recognition (MNIST).
    * :ref:`(Lecun 1998) <lecun1998gradient>` describes practical applications for CNNs.

lecun1989generalization
-----------------------
Handwritten digit recognition with a back-propagation network

Yann LeCun (Lecun 1989) proves that minimizing the number of free parameters in neural networks can enhance the generalization ability of neural networks.

Related: 
    * :ref:`(Lecun 1989) <lecun1989backpropagation>` is the original CNN paper.
    * :ref:`(Lecun 1989) <lecun1989handwritten>` describes practical applications of CNNs for handwritten digit recognition (MNIST).
    * :ref:`(Lecun 1998) <lecun1998gradient>` describres practical applications for CNNs.

lecun1989handwritten
--------------------
Handwritten digit recognition with a back-propagation network

(Lecun 1989) describes the application of backpropagation networks in handwritten digit recognition once again.

Related: 
    * :ref:`(Lecun 1989) <lecun1munoz2015m3gp989backpropagation>` is the original CNN paper.
    * :ref:`(Lecun 1989) <lecun1989generalization>` describes the generalization ability of CNNs.
    * :ref:`(Lecun 1998) <lecun1998gradient>` practical applications of LeNet. 

lecun1998gradient
-----------------
Gradient-based learning applied to document recognition

(Lecun 1998) shows the practical applications of LeNet for document recognition.

Related: 
    * :ref:`(Lecun 1989) <lecun1989backpropagation>` is the original CNN paper.
    * :ref:`(Lecun 1989) <lecun1989generalization>` describes the generalization ability of CNNs.
    * :ref:`(Lecun 1989) <lecun1989handwritten>` describes practical applications of CNNs for handwritten digit recognition (MNIST).

lee2019wide
-----------
    * Wide neural networks of any depth evolve as linear models under gradient descent

lehman2020surprising
--------------------
The surprising creativity of digital evolution: A collection of anecdotes from the evolutionary computation and artificial life research communities
 
(Lehman 2020) give annecdotes from researchs in EC about their algorithms demonstrating bizzare interesting behaviour. 

lensen2017new
-------------
New representations in genetic programming for feature construction in k-means clustering

(Lensen 2017) 

Related: 
    * Discussed in :ref:`2023-09-28 - FASLIP <2023-09-28 - FASLIP>`

lex2022noam
-----------
Noam Brown: AI vs Humans in Poker and Games of Strategic Negotiation | Lex Fridman Podcast #344

Notes: 
    * Counter-factural regret minimization (CFR) https://youtu.be/2oHH4aClJQs?t=951
    * Imperfect information games, e.g. poker, rock-paper-scissors, etc.
    * Litratus - latin for balance - how often to play each action. 
    * Elo rating system - https://en.wikipedia.org/wiki/Elo_rating_system
    * Top chess players have an Elo around 3,600.
    * Strongest version of AlphaZero is around 52,000 Elo.
    * If you remove search, forward-planning, Elo drops to 3,000.
    * Niether Libratus/Pluribus use neural nets, instead constrain the state-space search in a clever way! 
    * Diplomacy - natural lanaguage game that is similar to Civilisation. 
    * Action-state is near infinite.
    * Set in pre-war Europe, need to form alliances, goal to conquer the entire map (Europe).
    * Human-like, turing test - as humans gang up on bots when they find them (in-group preference?), implied that human-like behaviour is needed to win.
    * Fairness, humans kill teammates to seek retribution for unfiarness, even at the cost of winning, bots don't do this.
    * Very similar behaviour to Monkeys :ref:`(Brosnan 2003) <brosnan2003monkeys>`.

Available: https://youtu.be/2oHH4aClJQs

Related: 
    * :ref:`(Brown 2019) <brown2019superhuman>` Pluribus beats humans at 6 person no-limit Texas hold 'em poker
    * :ref:`(Brown 2018) <brown2018superhuman>` Libratus beats humans at heads-up no-limit Texas hold 'em poker.
    * :ref:`(Brown 2022) <brown2022human>` shows that AI can beat humans at diplomacy.
    * :ref:`(Morvavvcik 2017) <moravvcik2017deepstack>` DeepStack beats humans at heads-up no-limit Texas hold 'em poker.
    * :ref:`(Brosnan 2003) <brosnan2003monkeys>` monkeys reject unequal pay.

li2002novel
-----------
A novel evolutionary algorithm for determining unified creep damage constitutive equations

(Li 2002) use evolutionary computation to solve differentiral equations for deriving physics laws. 

Notes:
    * Creep behaviours of different materials are often described by physically based unified creep damage constitutive equations.
    * Such equations are extremely complex.
    * They often contain undecided constants (parameters).
    * Traditional approaches are unable to find good near optima for these parameters. 
    * Evolutionary algorithms (EAs) have been shown to be very effective.

Related: 
    * See :ref:`2022-11-10 - FASLIP<2022-11-10 - FASLIP>` where author Xin Yao discussed this paper. 
    * :ref:`(Li 2004) <li2004evolutionary>`, by  Xin Yao same author, with EC for solving DE in astrophysics. 
    * :ref:`(Runarsson 2000) <runarsson2000stochastic>` used stocastic ranking (bubblesort variant) for constrained optimization with Evolutionary Computaiton.
    * :ref:`(Handa 2006) <handa2006robust>`, by Xin Yao same author, use evolutionary computation for route optimization for gritting trucks. 

li2004evolutionary
------------------
An evolutionary approach to modeling radial brightness distributions in elliptical galaxies

(Li 2004) use evolutionary computation to find models that fit observational data in astrophysics.

Notes:
    * Empirical laws are widely used in astrophysics.
    * However, as the observational data increase, some of these laws do not seem to describe the data very well.
    * Can we discover new empirical laws that describe the data better?
    * Previous approach: 
        * Select a functional form in advance
        * Drawbacks: ad hoc, difficult to determine and may only suit a smaller number of profiles
        * Apply fitting algorithms to find suitable parameters for the function. Usually adopt the non-linear reduced c2 minimization
        * Drawbacks: difficult to set initial values and easily trapped in local minima
    * Proposed (Li 2004) evolutionary approach: 
        1. Find functional forms using GP (Genetic Programming) :
            * A data-driven process without assuming a functional form in advance
            * A bottom up process which suits modelling a large number of galaxy profiles without any prior knowledge of them
        2. Fit parameters in the form found using FEP (Fast Evolutionary Programming):
            * Not sensitive to initial setting values
            * More likely to find global minima

Related: 
    * See :ref:`2022-11-10 - FASLIP<2022-11-10 - FASLIP>` where author Xin Yao discussed this paper.
    * :ref:`(Li 2002) <li2002novel>`, Xin Yao same author, with EC for solving DE in materials science.
    * :ref:`(Runarsson 2000) <runarsson2000stochastic>`, Xin Yao same author, used stocastic ranking (bubblesort variant) for constrained optimization with Evolutionary Computaiton.
    * :ref:`(Handa 2006) <handa2006robust>`, by Xin Yao same author, use evolutionary computation for route optimization for gritting trucks.

li2017feature
-------------
Feature selection: A data perspective

(Li 2017) is a literature survey of feature selection algorithms.

Available: https://dl.acm.org/doi/abs/10.1145/3136625

Related: 
    * Discussed by Ruwang in :ref:`2023-12-08 - ECRG <2023-12-08 - ECRG>`

li2021learnable
---------------
Learnable fourier features for multi-dimensional spatial positional encoding

(Li 2021) propose a spatial encoding that works for multi-dimensional data, such as multi-modal input. 

Background: 
    * :ref:`(Vaswani 2017)<vaswani2017attention>` originally proposed sinosoidal positional encodings for 1D data.
        * This was the original transformers paper, with catchy title, "Attention is all you need".
    * Sinosoidal positional encodings are not suited for multi-dimensional or multi-modal data. 
    * Previous methods rely on hard-coding each position as a token or a vector.

Data: 
    * High volume structured data, such as video, images, audio, etc...
    * Multi-modal - inputs from different modalities, such as video, audio, text, etc...

Method: 
    * They propose positional encodings that use Learnable Fourier features
    * we represent each position:
        * which can be multi-dimensional, 
        * as a trainable encoding based on learnable Fourier feature mapping, 
        * modulated with a multi-layer perceptron

Why it matters?
    * Use positional encodings for multi-modal / multi-dimensional data. 
    * Useful for semi-supervised learning applications. 

Related: 
    * See :ref:`(Vaswani 2017) <vaswani2017attention>` proposed sinosoidal positional encodings for 1D data.
    * See :ref:`(Jaegle 2021) <jaegle2021perceiver>` for Perciever paper that uses these encodings. 
    * See :ref:`(Peng 2023) <peng2023rwkv>` for RWKV paper that builds on this. 
    * See :ref:`(Zhai 2021) <zhai2021attention>` for attention free transformer (AFT paper).
    * See :ref:`(Wang 2020) <wang2020linformer>` for Linformer paper.
    * See :ref:`(Kitaev 2020) <kitaev2020reformer>` for Reformer paper. 
    * See :ref:`(Katharopoulos 2020) <katharopoulos2020transformers>` for linear transformers. 

li2023blip
----------
Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models

(Li 2023) propose Blip-2 for multimodal pre-training

Available: https://arxiv.org/abs/2301.12597

Related: 
    * Discussed in :ref:`2024-02-29 - FASLIP <2024-02-29 - FASLIP>`

lin2017feature
--------------
    * Feature pyramid networks for object detection. 
    * Feature Pyramid Network (FPN)
    * See :ref:`2022-10-06 - FASLIP<2022-10-06 - FASLIP>`-  for more details.

linnainmaa1970representation
----------------------------
The representation of the cumulative rounding error of an algorithm as a Taylor expansion of the local rounding errors

(Linnainmaa 1970) proposed automatic differentiation (AD).

Notes: 
    * Backpropagation is a special case of reverse mode automatic differentiation. 
    * Reverse mode automatic differentiation is more efficient than forward mode automatic differentiation.
    * Requires on foward and one reverse pass to propogate error and adjusrt the weights through the network.
    * However, it requires more memory than forward mode automatic differentiation, as it needs to store the intermediate values of the forward pass.

Related:
    * Everything that uses backpropagation. 
    * All of deep learning. 
    * :ref:`(LeCun 1989) <lecun1989backpropagation>` is the original CNN paper.

liu1995chi2
-----------
    * Chi2: feature selection and discretization of numeric attributes
    * Discretization bins continuous values into discrete ones.  
    * Feature selection via discretization - ideal for numeric data.
    * Motivation: (1) (can) improve performance, (2) efficiency (time/space), (3) simplify models. 
    * Chi2 discretizes and performs FS - useful as many algorithms perform better with discrete/binary data. 
    * Under discretization would return the original continuous attribute unchanged. 
    * Over-discretization is when inconsistencies are introduced to the data - the data loses fidelity. 
    * Previous work, ChiMerge (Kerber 1992, kerber1992chimerge) with hyper-parameter :math:`\alpha` the significance level that had to be manually set. 
    * :math:`\alpha` is nuisance variable that requires black magic approach to tune.
    * Difficult to find ideal :math:`\alpha` without domain knowledge or extensive trial and error. 
    * New approach Chi2 lets data determine value of :math:`\alpha`, perform discretization until over-discretization - a stopping criterion. 
    * Chi2 is a two-phase method, a generalized version of ChiMege that automatically determines a good :math:`\chi^2` threshold that fits the data.
    * The formula for calcutaling the $\chi^2$ statistic is given by, :math:`\chi^2 = \sum_{i=1}^2 \sum_{j=1}^k \frac{(A_{ij} - E_{ij})^2}{E_{ij}}`.
    * Phase 1: Extends ChiMerge to be an automated one, to select an ideal value for :math:`\alpha` based on the data. 
    * Phase 2: Each feature is assigned signfnicance level and merged in a round robin fashion - until stopping criterion met. 
    * Attributes only merged to one value are elminianted as part of feature selection. 
    * Degrees of freedomlensen2017new: the maximum number of logically independent values, which are values that have the freedom to vary, :math:`D_F = N - 1`, where :math:`N =` samples, :math:`D_F =` degrees of freedom. 
    * If :math:`R_i` or :math:`C_i` is zero, set to 0.1. Similar to zero frequency problem from Naive Bayes. I.e. Multiplication by zero is always 0, so all other information is lost. 
    * Experiments: DT (C4.5), Data with Noise, and Synthetic data. 
    * Datasets: Iris (continious), Breat (discrete), Heart (mixed).
    * C4.5, a DT classification algorithm, is run on its default setting.
    * Results show predictive accuracy and size, same or improved for all datasets where Chi2 was applied.
    * Chi2 was able to remove noise (irrelvant features) from synthetic and real world data.

liu2018darts
------------
DARTS: Differentiable Architecture Search

(Liu 2018) propose DARTS, a differentiable architecture search algorithm.

Available: https://arxiv.org/abs/1806.09055

Related: 
    * Discussed in :ref:`2024-02-22 - FASLIP <2024-02-22 - FASLIP>`

liu2023instaflow
----------------
InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation

(Liu 2023) propose InstaFlow as a faster/cheaper one-step diffusion model for text-to-image generation.

Available: https://arxiv.org/abs/2309.06380

Related: 
    * Discussed in :ref:`2023-08-11 - Deep Learning <2023-08-11 - Deep Learning>`
    * DDPM :ref:`(Ho 2020) <ho2020denoising>`
    * DDIM :ref:`(Song 2020) <song2020denoising>`
    * Consistency models :ref:`(Song 2023) <song2023consistency>`
    * Diffusion borrows concepts from thermodyamics :ref:`(Sohl 2015) <sohl2014deep>`
    * Uses SDXL to improve resolution of generated output  :ref:`(Podel 2023) <podell2023sdxl>`
    
lock2007new
-----------
New Zealand's quota management system: a history of the first 20 years

(Lock 2007) gives a history of the New Zealand Quota Management System (QMS) for the first 20 years.

Related: 
    * :ref:`(batstone 1999) <batstone1999new>` give history of QMS in NZ for 10 years.

loh2011classification
---------------------
    * Decision tree. 

loshchilov2017decoupled
-----------------------
Decoupled weight decay regularization

(Loshchilov 2017) propose AdamW

Availbale: https://arxiv.org/abs/1711.05101

Notes:
    * "decoupling weight decay from the learning rate." - Christian Raymond 

mantyla1998cuelensen2017new
--------------
    * Cue distinctiveness and forgetting: Effectiveness of self-generated retrieval cues in delayed recall. 
    * Students were given a word list, and asked to make 1 or 3 retrieval cues. 
    * Students with who used their own multiple retrieval cues had better recall.
    * Recall was terrible when using another students own personal retrieval cues. 
    * Multiple self-generated retrieval cues is the most effective approach to maximising recall. 

marhsall2022cybermarine
-----------------------
Cyber-marine: 100 percent utilisation, maximised value

(Marshall 2022) from Cyber-marine gives an overview of their research aims on pg. 49 of Seafood New Zealand - Issue #226. 

TODO [ ] READ THIS 

Notes: 
    * Cybermarine research magazine aims. 

Related:
    * :ref:`(Wood 2016) <wood2022automated>` was colab between Cybermarine and VUW.

marine2020tackling
------------------
Tackling Seafood Fraud

(Marine 2020) is an article from the Marine Steward Council (MSC) on seafood fraud in New Zealand. 

TODO [ ] READ THIS!!!

Def. fish fraud: 
    Food fraud, simply put, is the selling of food products with a misleading label, description or promise.

Links: 
    * Available: https://www.msc.org/media-centre/news-opinion/news/2020/02/25/tackling-seafood-fraud 
    * Cool video: https://www.youtube.com/watch?v=Kac1cqkjX1U

Related: 
    * :ref:`(Pardo 2016) <pardo2016misdescription>` 30% of seafood is mislabelled.
    * :ref:`(Black 2017) <black2017real>` REIMS for fish fraud detection. 
    * :ref:`(Wood 2022) <wood2022automated>` fish speciation with Gas Chromatography.

matyushin2020gas
----------------
    * Matyshuin et al. proposed a stacking model for analysis of gas-chromatograph data.
    * It stacked the results of 1DConv, 2DConv, Deep Residual MLP and XGBoost.
    * Their model predicted the retention index for samples.
    * A retention index is a standardized value that only depends on the chemical structure of a compound.
    * Once identified the retention index can be used for further identification.
    * GC-MS data has underlying patterns that correspond to chemical compounds.

mccann2012local
---------------
Local naive bayes nearest neighbor for image classification

`(McCann 2012) <https://ieeexplore.ieee.org/abstract/document/6248111>` propose a local naive bayes nearest neighbor (LNBNN) classifier.

Related: 
    * :ref:`(Crall 2013) <crall2013hotspotter>` uses LNBNN for instance recognition.
    * :ref:`(Behmo 2010) <behmo2010towards>` proposed Naive Bayes Nearest Neighbor (NBNN).

mcconnell1986method
-------------------
Method of and apparatus for pattern recognition

(McConnell 1986) proposed Histograms of Oriented Gradients (HOG) for pattern recognition.

Available: https://www.osti.gov/biblio/6007283

Related: 
    * Discussed in :ref:`2023-10-12 FASLIP <2023-10-12 FASLIP>`

mclean2005differences
---------------------
Differences in lipid profile of New Zealand marine species over four seasons

(Mclean 2005) describes the seasonal-variation for lipids in Hoki fish when spawning.

Related: 
    * :ref:`(Sun 2022)<sun2022soknl>` for concept drift from data steam mining. 
    * See :ref:`(2023-02-16 - FASLIP)<2023-02-16 - FASLIP>` for talk from (Sun 2022) author.
    * :ref:`(Gomes 2022)<gomes2020ensemble>` for concept drift from data stream mining.
    * See (Wood 2023), my proposal, which references Hoki seasonal variation.

mikolov2013efficient
---------------------
Efficient Estimation of Word Representations in Vector Space

(Mikolov 2013) proposed word2vec.

Related; 
    * Cited in :ref:`2023-10-06 - ECRG <2023-10-06 - ECRG>`
    * Same author of :ref:`(Mikolov 2013) <mikolov2013linguistic>`.

mikolov2013linguistic
---------------------
Linguistic Regularities in Continuous Space Word Representations

(Mikolov 2013) found semantically meaningful feature embeddings for natural language, e.g. "King" - "Man" + "Woman" = "Queen"

Notes: 
    * Mikolov et al. found the word embeddings used in NLP were semantically meaningful \cite{mikolov2013linguistic}. 
    * They showed arithmetic could be applied to these word vectors that were interpretable. 
    * For example "King" - "Man" + "Woman" = "Queen". 
    * The feature space was semantically meaningful, which serves as a powerful representation, that we intuitively reason with. 
    * Similar thought has been applied to computer vision \cite{olah2018building, karras2020analyzing}. 
    * Semantically meaningful feature spaces allow for intuition about the behaviour of complex models, be it through visualisation or arithmetic.

Related: 
    * Related to node2vec (Grover 2016) for graph embeddings.
    * Related to (Olah 2018) for feature visualisation.
    * Same author as :ref:`(Mikolov 2013) <mikolov2013efficient>`.

miles1998state
--------------
    * State-dependent memory produced by aeorobic exercise. 
    * Students studies while exercising on a treadmil. 
    * Material learnt on the treadmill was better recalled on the treadmill. 
    * Greater information retrieval when the state (i.e. aerobic exercise) is similar. 

miller1994exploiting
--------------------
    * Complement natural selection with sexual selection. 
    * Biological theory behind sexual selection. 
    * Sexual selections influences culture around metrics for fitness/fertility. 
    * Gendered candidate solutions. 
    * Mate choice / mate preference. 
    * **TODO** read 

miller2017explainable
---------------------
Explainable AI: Beware of inmates running the asylum or: How I learnt to stop worrying and love the social and behavioural sciences},

(Miller 2017) talks about the pitfalls of academics defining XAI. 

Related: lensen2017new
    * :ref:`(Miller 2021) <miller2021contrastive>` contrastive explanation.
    * :ref:`(Miller 2019) <miller2019explanation>` explanation in AI.
    * :ref:`2022-12-07 - AJCAI<2022-12-07 - AJCAI>` for talk from author.

miller2019explanation
---------------------
Explanation in artificial intelligence: Insights from the social sciences},

(Millier 2019) addresses the disconnect between XAI and social sciences. 

Related: 
    * :ref:`(Miller 2021) <miller2021contrastive>` contrastive explanation.
    * :ref:`(Miller 2017) <miller2017explainable>` explainable AI.
    * :ref:`2022-12-07 - AJCAI<2022-12-07 - AJCAI>` for talk from author.

miller2021contrastive
---------------------
Contrastive explanation: A structural-model approach

(Miller 2021) proposes a new approach to explainable AI, called contrastive explanation.

Related:
    * :ref:`(Miller 2017) <miller2017explainable>` explainable AI.
    * :ref:`(Miller 2019) <miller2019explanation>` explanation in AI.
    * :ref:`2022-12-07 - AJCAI<2022-12-07 - AJCAI>` for talk from author.

morgan1989generalization
------------------------
Generalization and parameter estimation in feedforward nets: Some experiments

(Morgan 1989) propose early stopping for neural networks.

Available: https://proceedings.neurips.cc/paper_files/paper/1989/hash/63923f49e5241343aa7acb6a06a751e7-Abstract.html

mnih2013playing
---------------
Playing atari with deep reinforcement learning

(Mnih 2013) from Deep Mind propose deep q-learning for Atari games.

Related: 
    * 2022-12-05 - AJCAI #01

moraglio2012geometric
---------------------
    * Genetic semantic genetic programming. 
    * **TODO** read - related to Qi Chen talk on 2022-03-18 ECRG. 
    * Unimodal fitness landscape, one global optima, but semantic search is intractable. 
    * We approximate semantic search through geometric genetic programming methods. 

moravvcik2017deepstack
----------------------
Deepstack: Expert-level artificial intelligence in heads-up no-limit poker

(Moravcik 2017) shows that AI can beat human professionals at two-player no-limit Texas hold 'em poker.

DeepStack: DeepStack, an AI system that was able to consistently beat human professionals at two-player no-limit Texas hold 'em poker.

The research paper describing DeepStack was published in the journal Science in 2016 and can be found here: https://www.science.org/doi/full/10.1126/science.aam6960

Related: 
    * :ref:`(Brown 2019) <brown2019superhuman>` Pluribus beats humans at 6 person no-limit Texas hold 'em poker
    * :ref:`(Brown 2018) <brown2018superhuman>` Libratus beats humans at heads-up no-limit Texas hold 'em poker.
    * :ref:`(Morvavvcik 2017) <moravvcik2017deepstack>` DeepStack beats humans at heads-up no-limit Texas hold 'em poker.

mouret2015illuminating
----------------------
Illuminating search spaces by mapping elites

Related: 
    * (Hengzhe 2023) his GECCO 2023 paper uses MAP-elites in the semantic space. 
    * See 2023-02-10 - ECRG where Hengzhe discussed this paper, and his work above.

mouss2004test
-------------
Test of page-hinckley, an approach for fault detection in an agro-alimentary production system

Related: 
    * See (:ref:`Gomes 2020<gomes2020ensemble>`) for a paper that cites this. 
    * See :ref:`2023-02-16 - FASLIP<2023-02-16 - FASLIP>` where this method is mentioned.
    * See (:ref:`Sun 2022<sun2022soknl>`) for paper on data stream mining.

muller2021transformers
----------------------
    * Transformers Can Do Bayesian Inference
    * **TODO** read 
    * Transformers can do Bayesian inference, The propose prior-data fitted networks (PFNs). PFNs leverage large-scale machine learning techniques to approximate a larget set of posteriors (Muller 2021, muller2021transformers).
    * Requires the ability to sample from a prior distribution over supverised learning tasks (or functions). 
    * Their method restates the objective prosterior apprimixation as a supervised classification problem with set valued input: it repeatedly draws a task (or function) from the prior, draws a set of data points and their labels from it, marks on of the labels and learns to make probabilistic predictions for it based on the set-valued input of the rest of the data points.
    * PFNs can nearly perfectly mimic Gaussian Processes and also enable efficient Bayesian Inference for intractable problems, with 200-fold speedups in networks evaluated. 
    * PFNs perofrm well in GP regression, Bayesian NNs, classification on tabular data, few-shot iamge classification - there applications demonstrate generality of PFNs. 

munoz2015m3gp
-------------
M3GP – Multiclass Classification with GP

Available: https://link.springer.com/chapter/10.1007/978-3-319-16501-1_7

Notes: 


Related: 
    * Discussed in :ref:`2023-09-28 - FASLIP <2023-09-28 - FASLIP>`
    * Variation of M2GP from :ref:`(Ingalalli 2014) <ingalalli2014multi>`

nickerson2022creating
---------------------
Creating Diverse Ensembles for Classification with Genetic Programming and Neuro-MAP-Elites
   
* TODO [ ] - READ 
    
Related: 
    * Hengzhe is working on MAP-Elites in GP. See :ref:`2022-10-13 - FASLIP<2022-10-13 - FASLIP>`

nielsen2020survae
-----------------
    * SurVAE Flows: Surjections to Bridge the Gap between VAEs and Flows 
    * TODO [ ] read 

nguyen2014filtermccann2012local
----------------
    * Nguyen et al. proposed a wrapper based PSO technique for feature selection in classification.
    * The algorithm uses a wrapper based fitness function of the classification error rate.
    * The local search only considers the global best using a filter based method.
    * It draws from the strengths of filter and wrapper based feature selection.
    * This proposed method outperformed three state-of-the-art and two traditional feature selection methods.

olah2018building
----------------
The Building Blocks of Interpretability

(Olah 2018) from Distill shows how to visualise semantically meaningful features in computer vision.

Available: https://distill.pub/2018/building-blocks/?translate=1&translate=1&translate=1&translate=1&student&student&student&student

Notes: 
    * Semantically meaningful features in computer vision. 
    * Distill https://distill.pub/2018/building-blocks/
    * Visualization techniques are powerful for understanding black-box systems.
    * Gain intution for semantically meaninful features in complex models. 

Related: 
    * Original CNN paper :ref:`(Lecun 1989) <lecun1989backpropagation>`
    * word2vec, semantically meaningful feature embeddings for natural language :ref:`(Mikolov 2013) <mikolov2013linguistic>`
    * node2vec, feature embeddings for graphs :ref:`(Grover 2016) <grover2016node2vec>` 

pardo2016misdescription
-----------------------
Misdescription incidents in seafood sector

Highlights: 
* The average percentage of reported misdescription is 30%.
* Misdescription incidents are significantly greater in restaurants than retailers.
* Gadoids, flatfish and salmonids comprise almost the 60% of the total.
* Future surveys should be focused on other commercial species.

Method: 
    * DNA testing, good for species identification
    * compares 51 studies with total n=4,500 seafood samples. 

Results: 
    * found an average mislabelling rate of 30%

Related: 
    * :ref:`(Black 2017) <black2017real>` REIMS for fish fraud detection. 
    * :ref:`(Marine 2020) <marine2020tackling>` for fish fraud definition. 
    * :ref:`(Black 2019) <black2019rapid>` discusses DNA methods for speciation. 

pascual2022fullband
-------------------
Full-band General Audio Synthesis with Score-based Diffusion

Linked: 
    * Website https://diffusionaudiosynthesis.github.io/ 
    * ArVix https://arxiv.org/abs/2210.14661
    * Video https://twitter.com/_akhaliq/status/1585431732916027392

Related: 
    * :ref:`(Song 2020) <song2020denoising>` DDPM. 
    * :ref:`(Ho 2020) <ho2020denoising>` DDIM. 

pearce2021empirical
-------------------
    * 70% accuracy for basic DSA problems. 
    * Can't solve more difficult problems - doesn't optimize solutions for performance. 
    * CoPilot outperforms other state-of-the-art NLP code generation models. 
    * Requires "fine-tuning", supervised human intervention to hint towards correct answer. 

peng2022prenastvec
--------------
PRE-NAS: Evolutionary Neural Architecture Search with Predictor. IEEE Transactions on Evolutionary Computation.

(Peng 2022) is a IEEE TVEC paper on Pre-NAS (first paper).

Related:
    * See ECRG - 2023-01-20 for talk from author.
    * See :ref:`(Peng 2022) <peng2022prenastvec>` for GECCO paper, published later.

Peng2021prenasgecco
-------------------
PRE-NAS: Evolutionary Neural Architecture Search with Predictor. IEEE Transactions on Evolutionary Computation.

(Peng 2022) is a GECCO paper on Pre-NAS (second paper).

Related: 
    * See ECRG - 2023-01-20 for talk from author.
    * See :ref:`(Peng 2022) <peng2022prenastvec>` for TVEC paper, published earlier.

peng2023rwkv
------------
RWKV: Reinventing RNNs for the Transformer Era

`(Peng 2023) <https://arxiv.org/abs/2305.13048>`__ propose Receptance Weighted Key Value (RWKV). 

Notes: 
    * A hybrid model of transformers and RNNs.
    * More efficient inference than regular GPT-based transformers. 
    * Don't need GPU clusters to fine-tune - works on commodity hardware. 


Background:
    * The computation graph of a transformer is setup in a way that isi is very efficient to be computed in parallel on GPU clusters. 
    * (Zhai 2021) proposed the Attention Free Transformer (AFT), the predecessor to today's paper, the Receptance Weighted Key Value (RWKV) transformer. 

Method:devlin2018bert
    * RWKV Raven 14B Demo - Hugging Face https://huggingface.co/spaces/BlinkDL/ChatRWKV-gradio
    * Github https://github.com/BlinkDL/RWKV-LM
    
Related:
    * See :ref:`2023-05-10 - Deep Learning <2023-05-10 - Deep Learning>` where paper was dicussed.
    * See :ref:`(Zhai 2021) <zhai2021attention>` for Attention Free Transformer (AFT).
    * See :ref:`(Wang 2020) <wang2020linformer>` for Linformer paper.
    * See :ref:`(Kitaev 2020) <kitaev2020reformer>` for Reformer paper. 
    * See :ref:`(Katharopoulos 2020) <katharopoulos2020transformers>` for linear transformers. 
    * See :ref:`(Vaswani 2017) <vaswani2017attention>` for transformer paper.
    
perez2019analysis
-----------------
Analysis of statistical forward planning methods in Pommerman

Related: 
    * :ref:`(Volz 2018) <volz2018evolving>` same author evolves Mario levels using EAs on GAN latent spaces. 
    * :ref:`(Goodman 2020) <goodman2020weighting>` same user uses NBTEA to choose hyperparameters for balancing gamemplay.

podell2023sdxl
--------------
Sdxl: Improving latent diffusion models for high-resolution image synthesis

(Podel 2023) propose SDXL for scaling up images to high resolution.

Available: https://arxiv.org/abs/2307.01952

Related: 
    * Discussed in :ref:`2023-08-11 Deep Learning <2023-08-11 Deep Learning>`
    * Used in InstaFlow :ref:`(Liu 2023) <lui2023instaflow>`
    * Diffusion process based on thermodyamics :ref:`(Sohl-Diskstein 2015) <sohl2014deep>`

qin2021one
----------
    * From One Hand to Multiple Hands: Imitation Learning for Dexterous Manipulation from Single-Camera Teleoperation. 
    * TODO read 
    * This paper shows single-camera teleoperation capabilities for SCARA. 
    * This could be used to allow for remote intervention in edge cases for our SCARA. 

radford2018improving
--------------------
Improving language understanding by generative pre-training

Notes: 
    * Generative pre-training (GPT) paper from OpenAI. 

Available: https://www.mikecaptain.com/resources/pdf/GPT-1.pdf

Related: 
    * :ref:`2023-02-22 - Deep Learning <2023-02-22 - Deep Learning>` describes this paper. 
    * Same author, CLIP - :ref:`(Radford 2021) <radford2021learning>`
    * Same author, CLIP - :ref:`(Radford 2021) <radford2021zero>`

radford2021learning
-------------------
Learning Transferable Visual Models From Natural Language Supervision

(Radford 2021) propose CLIP a representational model for text-to-image feature embeddings.

Available: http://proceedings.mlr.press/v139/radford21a

Related: 
    * Same author, same model, same year :ref:`(Radford 2021) <radford2021zero>`
    * Same author :ref:`(Radford 2028) <radford2018improving>`

radford2021zero
---------------
Zero-Shot Text-to-Image Generation

(Radford 2021) propose CLIP a representational model for text-to-image feature embeddings.

Available: https://proceedings.mlr.press/v139/ramesh21a.html?ref=journey-matters

Related: 
    * Same author, same model, same year :ref:`(Radford 2021) <radford2021learning>`
    * Same author :ref:`(Radford 2028) <radford2018improving>`

rajpurkar2017chexnet
---------------------
Chexnet: Radiologist-level pneumonia detection on chest x-rays with deep learning

(Rajpurkar 2017) demonstrate leaky validation in medical x-ray images for computer vision.

Available: https://arxiv.org/abs/1711.05225

Background:
    * In 2017, Andrew Ng's team published a paper on Deep Learning for pneumonia detection on chest X-rays.
    
Dataset:
    * They used a dataset with 112,120 images belonging to 30,805 unique patients. They automatically labelled every sample with 14 different pathologies and randomly split the dataset into 80% training and 20% validation. Their process downscaled images to 224x224 pixels before inputting them into a neural network.
    
Limitations:
    * After publishing the paper and listening to the community's feedback, they had to redo their experiments.
    * The samples in the team's dataset are not independent. Different X-Ray images from the same patient will have similarities that a neural network could use to make a prediction.
    * For example, a patient might have a scar from a previous surgery or a specific bone density or structure. These clues will help the model make a prediction, so having X-rays from the same patient in the training and validation sets will create a leaky validation strategy. Here is an excerpt from The Kaggle Book:
    * "In a leaky validation strategy, the problem is that you have arranged your validation strategy in a way that favours better validation scores because some information leaks from the training data." (Kaggle)
    
Method:
    * The team fixed the experiment in the third version of their paper. Here is what they did:
    * "For the pneumonia detection task, we randomly split the dataset into training (28744 patients, 98637 images), validation (1672 patients, 6351 images), and test (389 patients, 420 images). There is no patient overlap between the sets." (Kaggle)
    * Notice how they ensured that there was no overlap between sets.

Conclusion:
    * Leaky validation is an issue with identifiable instances, leaking information from train to validation, that artificially inflate validation accuracy.
    * Leaky validation is a unique form of data leakage, datasets with identifiable features, such as medical/chemical/biological, are vulnerable to validation leakage.

Why it matters? 
    * For my research objectives of Identification, Contamination, and Traceability - the same fish in train and validation, is leaky validation.
    * Leaky validation explains why my CNN train/validation performance is 100%, but the test is 60%.
    * Lesson: stratify samples to avoid identifiable instances being leaked between train/validation/test datasets.

Related: 
   * The Kaggle Book https://www.amazon.com/Data-Analysis-Machine-Learning-Kaggle/dp/1801817472?crid=2ZSVOUZJCXMO5&keywords=kaggle+book&qid=1650818962&sprefix=kaggle+book,aps,72&sr=8-3&linkCode=sl1&tag=bnomial-20&linkId=6cf9fd66daf5893153a64f03302971f7&language=en_US&ref_=as_li_ss_tl
   * "Target Leakage in Machine Learning" is a YouTube presentation that covers leakage, including during the partitioning of a dataset. https://www.youtube.com/watch?v=dWhdWxgt5SU 

raine1997brain
--------------
    * Murderers pleading not guilty by reason of insanity (NGRI).
    * Pre-disposition to less activity in their pre-frontal cortex. 
    * Pre-frontal cortex associated with goal-directed planning and delayed gratification. 
    * Different brain chemistry meant more likely to perform violent impulsive behaviour. 
    * Justification for a lobotomy - electrocution of pre-frontal cortex - now replaced by antipsychotics. 

raissi2019physics
-----------------
    * Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations
    * Discussed by Bastiaan from :ref:`2022-09-14 - Deep Learning<2022-09-14 - Deep Learning>`  

ramesh2022hierarchical
----------------------
Hierarchical Text-Conditional Image Generation with CLIP Latents. 

Notes:
    * a.k.a. Dalle 2 and Very popular on the internet.
    * Original was a d-VAE (discrete), Dalle 2 is a diffusion based model that uses CLIP. 
    * CLIP trains an auto-enocder to have minimize the distance between image and text embeddings in the latent space. 
    * Those image embeddings are fed to an autoregressive or diffusion prior to generate image embeddings. 
    * Then this embedding is used to condition a diffusion decoder which produces an image. 
    * The model is trained on 250 Million images, and has 3.5 billion parameters. 
    * We can use CLIP to interpolate between two images in the latent space. 
    * As we increase the dimensionality of the latent space we can represent more complex hierarchical structures. 
    * CLIP fails at producing text, and reconstruction can mix up objects and their attributes. 

Related: 
    * Available to the public on their website https://openai.com/dall-e-2/
    * See :ref:`(Ho 2022) <ho2020denoising>` for Denoising Diffusion Probabilistic Models (DDPM), canonical diffusion paper. 
    * See :ref:`(Song 2020) <song2020denoising>` for Denoising Diffusion Implicit Models (DDIM), faster diffusion process.
    * See :ref:`2022-07-06 - Deep Learning<2022-07-06 - Deep Learning>`
    * See :ref:`2022-10-19 - Deep Learning<2022-10-19 - Deep Learning>` 
    * See :ref:`2023-05-03 - Deep Learning <2023-05-03 - Deep Learning>`

rampal2022high
--------------
High-resolution downscaling with interpretable deep learning: Rainfall extremes over New Zealand

(Rampal 2020) propose CNN rainfall downscaling for prediction of extreme rainfall. 

Available https://www.sciencedirect.com/science/article/pii/S2212094722001049

Data: 
    * Daily gridded accymlated ranfall from the Viritual Climatete Station network (VCSN), was used here as the predicted rainfall in statistical downscaling, and as the ground truth in out-of-sample testing period. 
    * VCSN data covers the New Zealand regoion on a 0.05 degree grid, derived from sufrace interpolation of station weather data. 
    * Precipitation biases are likely to occur in regions where station density is particularly low, namely across rugged and remote terrian like the Southern Alps. 

Method: 
    * The model trains predictor fields, the variables represent both dynamical and thermodynamical drivers of rainfall. 
    * Loss functions of Mean Squared Error (MSE) and log-likelihood of Bernoilli-gamma distribution. 
    * 5 deep learning achrictures were evaluated
        1. Non-linear CNN (Gamma)
        2. Linear CNN 
        3. Non-linear Gamma 
        4. Linear Gamma 
        5. Linear Dense
    * Explainabile AI, deep learning models are often referred to as black box models. 
    * Gradient-weighted class-activation maps (Grad-Cam)
    * Alternative methods, salience maps and layerwise relevance propogation were considered.
        * Saliency maps:
            * but don't necessarily imply importance. Instead, they show how the output changes when a set of predicotr values are slightly perturbed. 
            * Saliency maps also tend to focus on local gradients in the input spoace whereas a more global view if odten required.
        * Layerwise relevance propogation (LRP):
            * LRP is only available for a relatively small set of neural entwrok architectures. 
            * Grad-CAM can be applied more widely. 

Results: 
    * When spatially aggregates across the region, the fraciton of explained vaiation on wet days increased from 0.35 to 0.52. The existing dry bais for rainfall extremes decreased from approximately 40% to 15%. 
    * Largest benefits came from implementing a probablistic loss function. Further improvements come from convolutional layers and non-linear activations. 
    * Non-linear CNN is capable of outperforming existing statistical approaches, both in terms of variance and mean predictions for extreme rainfall.
    * The trained CNN could target the most relevant meteorological features. Suggests the model is capable of learning complex and physically plausible relationships. 
    * Increasing the domain size over which predictor fields are sampled and increasing the number of training samples generally improves out-of-sample donwscaling performance. The domain size had less effect on linear models (i.e. low variance), sugggesting linear CNNs are better suited for extracting complex information across an extended domain. 

Why it matters? 
    * Simple CNN models can easily outperform statistical models on high-dimensional data.
    * Deep learning can be analyzed post-hoc, to build trust in the prediction, or may even lead to new insights.
    * Interpretable models are important bo build trust in their predictions. Also for troubleshooting/diagnosi, as in :ref:`(Zhao 2019 <zhao2019maximum>`.

Related:
    * :ref:`(Lecun 1989) <lecun1989generalization>` propsoed the original CNN as a shared weight network. 
    * :ref:`(Wang 2018) <wang2018evolving>` proposed EvoCNN, uses variable length PSO to perform neural architecture search. 
    * :ref:`(Girsich 2014) <girshick2014rich>` proposed R-CNN, a CNN with region proposals.
    * :ref:`(Bi 2020) <bi2020gc>` used CNN to predict food flavor from GS-MS datasets. 

rasmussen2003gaussian
---------------------
    * Gaussian Processes in machine learning. 

restek2018high
--------------
    * Explanation of gas-chromatraphy in food science for FAMEs. 

riad2022learning
----------------
    * Learning strides in convolutional neural networks 

riccardo2009field
-----------------
    * A Field Guide to Genetic Programming
    * A free resource for GP research available online. 

robinson2020genetic
-------------------
    * Demelza et al. proposed a feature and latent variable selection method for regression models in food science.
    * The vibrational spectroscopy dataset shared similarities in its high dimensionality and food science domain.
    * The purposes GA-PLSR generalized better and produced fewer complex models.
    * The study showed that Genetic Algorithms are powerful tools for feature selection in food science.

robnik2003theoretical
---------------------
    * releifF classifier. 

runarsson2000stochastic
-----------------------
Stochastic ranking for constrained evolutionary optimization

(Runarsson 2000) used stocastic ranking (bubblesort variant) for constrained optimization with Evolutionary Computaiton.

Notes:
    * Real-world problem has many constraints, e.g., linear, nonlinear, equality, inequality, ...
    * It works better than other methods because 
        * More effective;
        * Good at dealing with non-differentiable
        and nonlinear problems;
        * Avoid unnecessary and unrealistic
        assumptions.
    * Stochastic Ranking
        * It is a simple yet effective constraint
        handling method.
        * It exploits the characteristics of
        evolutionary algorithms.

Related: 
    * :ref:`(Li 2002) <li2002novel>`, by Xin Yao same author, use evolutionary computation to solve differentiral equations for deriving physics laws. 
    * :ref:`(Li 2002) <li2002novel>`, by Xin Yao same author, with EC for solving DE in materials science.
    * :ref:`(Handa 2006) <handa2006robust>`, by Xin Yao same author, use evolutionary computation for route optimization for gritting trucks. 
    * :ref:`(Schnier 2004) <schnier2004digital>`, by Xin Yao same author, use evolutionary computation for multi-objective optimisation in computer hardware. 


russell2010artificial
---------------------
Artificial intelligence a modern approach

(Russell 2010) is the phat textbook I own on AI. 

shahriari2015taking
-------------------
    * Taking the Human Out of theLoop: A Review of Bayesian Optimization.
    * Recommended reading from the :ref:`2023-03-24 - FASLIP<2023-03-24 - FASLIP>` on Bayesian Optimization
    * **TODO** read this. 

schnier2004digital
------------------
Digital filter design using multiple pareto fronts

(Schnier 2004) use evolutionary computation for multi-objective optimisation in computer hardware. 

Related: 
    * :ref:`(Li 2002) <li2002novel>`, by Xin Yao same author, use evolutionary computation to solve differentiral equations for deriving physics laws. 
    * :ref:`(Li 2002) <li2002novel>`, by Xin Yao same author is another paper by same author, with EC for solving DE in materials science.
    * :ref:`(Runarsson 2000) <runarsson2000stochastic>`, by Xin Yao same author, used stocastic ranking (bubblesort variant) for constrained optimization with Evolutionary Computaiton.
    * :ref:`(Handa 2006) <handa2006robust>`, by Xin Yao same author, use evolutionary computation for route optimization for gritting trucks. 

scholkopf2000new
----------------
    * Nu-SVC classifier. 
    * Setting the number of support vectors is a hyper-parameter.
    * Usually this is learned by the system. 

shaukat2022state
----------------
A state-of-the-art technique to perform cloud-based semantic segmentation using deep learning 3D U-Net architecture

(Shaukat 2022) use Dice pixel classification layer, a loss function for imbalanced datasets.

Available: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-04794-9

Related: 
    * :ref:`2023-11-30 FASLIP <2023-11-30 FASLIP>`
    * Auhtor of :ref:`(Emrah 2022) <emrah2022imbalance>` uses Dice loss for imbalanced data.

schulman2017proximal
--------------------
Proximal Policy Optimization Algorithms

(Schulman 2017) propose Proximal Policy Optimization (PPO) for reinforcement learning.

Available: https://arxiv.org/abs/1707.06347

Related: 
    * :ref:`2023-09-04 - Deep Learning <2023-09-04 - Deep Learning>`

simonyan2014very
----------------
Very deep convolutional networks for large-scale image recognition

(Simonyan 2014) is the original VGG paper.

Related: 
    * :ref:`(Lecun 1989) <lecun1989backpropagation>` proposed LeNet, the original CNN.
    * :ref:`(Krizhevsky 2012) <krizhevsky2012imagenet>` proposed AlexNet, the first CNN to win ImageNet.
    * :ref:`(He 2016) <he2016deep>` proposed ResNet, a CNN with residual connections.
    * :ref:`(Szegedy 2015) <szegedy2015going>` proposed GoogLeNet, a CNN with inception modules.
    * :ref:`(Huang 2017) <huang2017densely>` proposed DenseNet, a CNN with dense connections.

smart2005using
--------------
Genetic programming for multiclass object classification.

(Smart 2005) describe classification maps as a method for mutli-class classification using GP. 

Notes: 
    * Using genetic programming for multiclass classification by simultaneously solving component binary classification problems 
    * Multi-class classification with Genetic Programs using a Classification Map (CM). 
    * Maps a float to a classification label using a classification map.
    * Create class boundaries sequentially on a floating point number line. 
    * If program output is within a class boundary, it belongs to that class. 
    * For multi-class classification, their is an identical interval of 1.0. 

Related:
    * See proposal for preliminary work section, where classification maps are used. 

sobel1990isotropic
------------------
An Isotropic 3x3 Image Gradient Operator

Available: https://www.researchgate.net/publication/239398674_An_Isotropic_3x3_Image_Gradient_Operator

Related: 
    * Discussed in :ref:`2023-10-12 - FASLIP <2023-10-12 - FASLIP>`
    * See history of Sobel operator :ref:`(Sobel 2014) <sobel2014history>`

sobel2014history
----------------
History and Definition of the so-called "Sobel Operator",more appropriately named th eSobel-Feldman Operator

Available: https://www.researchgate.net/publication/239398674_An_Isotropic_3x3_Image_Gradient_Operator

Related:
    * Discussed in :ref:`2023-10-12 - FASLIP <2023-10-12 - FASLIP>`
    * See history of Sobel operator :ref:`(Sobel 2014) <sobel2014history>`


sobieczky1999parametric
-----------------------
Parametric Airfoils and Wings

(Sobieczky 1999) propose the PARSEC parametric airfoil and wing design system.

Notes: 
    * Parameteric model that uses 11 or 12 parameters to repreent major structural sectional features of an airfoil. 
    * Including:   
        * leading edge radii,
        * upper and lower crest location,
    * Constructs an airfoil using a sixth-order polynomial.

Available: https://link.springer.com/chapter/10.1007/978-3-322-89952-1_4

Related: 
    * This real-world task is discussed in :ref:`2023-09-07 - FASLIP <2023-09-07 - FASLIP>`

sohl2015deep
------------
Deep unsupervised learning using nonequilibrium thermodynamics

(Sohl 2015) borrow ideas from thermodynamics denoising autoencoders. 

Available: http://proceedings.mlr.press/v37/sohl-dickstein15.html

Related: 
    * Discussed in :ref:`2023-08-11 - Deep Learning <2023-08-11 - Deep Learning>`
    * Used in InstaFlow :ref:`(Liu 2023) <lui2023instaflow>`
    * DDPM :ref:`(Ho 2022) <ho2020denoising>`
    * DDIM :ref:`(Song 2020) <song2020denoising>`
    * Consistency models :ref:`(Song 2023) <song2023consistency>`

song2020denoising
-----------------
Denoising diffusion implicit models. 

(Song 2020) propose Denoising Diffusion Implicit Models (DDIM) a generalized DDPM that is faster and deterministic. 

Notes: 
    * TODO [ ] Read this paper! 

Related: 
    * See :ref:`(Ho 2022) <ho2020denoising>` for original DDPM paper. 
    * See :ref:`2022-07-06 - Deep Learning<2022-07-06 - Deep Learning >` 
    * See :ref:`2022-10-19 - Deep Learning<2022-10-19 - Deep Learning>` 
    * Stable Diffusion https://github.com/CompVis/stable-diffusion
    * Deforum Notebook https://t.co/mWNkzWtPsK

song2023consistency
-------------------
Consistency Models 

`(Song 2023) <https://arxiv.org/abs/2303.01469>`__ proposes consistency models, a faster alternative to diffusion. [Available] https://arxiv.org/abs/2303.01469

Background: 
    * OpenAI paper, so it's a big deal.
    * OpenAI, ChatGPT, GPT-3/4/5, Bing, DALLE-2 

Motivations:
    * inpainting, colorization, and super-resolution
    * inpainting: remove objects, fill in missing pixels
    * photograph-to-drawing (and vice versa)

Data: 
    * Trained as either:
        1. distilling an existing pre-trained DM
        2. standalone generative models

Method: 
    * The math in figure 1 & 2 https://twitter.com/jrhwood/status/1653620236145528833?s=20
    * Artificial noise is added to pictures. 
    * Consistency models learn to map and reverse that noise process, 
    * track the ODE trajectory back to its origin, 
    * i.e. the denoised original input image.

Results: 
    * See figure 12 https://twitter.com/jrhwood/status/1653625255238459394?s=20
    * (top) the same input image with different levels of noise, and,
    * (bottom) the output of the consistency model.
    * The model *consistently* denoised very similar images, that closely resemble the original input image, and each other.

Why it matters? 
    * Consistency models are based on diffusion, but cheaper for inference and at least as good as SOTA.
    * Explicit fast one-step generation by design.
    * (Optional) few-step sampling, improves quality with more compute needed.
    * Zero-shot editing without explicit training.

Limitations: 
    * Text information is not encoded in the embedding space. 

Related: 
    * See :ref:`(Ho 2022) <ho2020denoising>` for original DDPM paper. 
    * See :ref:`2022-07-06 - Deep Learning<2022-07-06 - Deep Learning >` 
    * See :ref:`2022-10-19 - Deep Learning<2022-10-19 - Deep Learning>` 
    * Stable Diffusion https://github.com/CompVis/stable-diffusion
    * Deforum Notebook https://t.co/mWNkzWtPsK

stewart2022quarry
-----------------
QUARRY: A Graph Model for Queryable Association Rules

(Stewart 2022) propose QUARRY a model for association rule mining from short technical text in maintenance data.

Related: 
    * See 2022-12-05 - AJCAI #01, author gave workshop on knowledge graphs. 

srivastava2014dropout
---------------------

Dropout: a simple way to prevent neural networks from overfitting

(Srivastava 2014) propose dropout for regularization in neural networks.

Available: https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf

Related: 
    * Also about regularization :ref:`(Hinton 2012) <hinton2012improving>`


sun2022soknl
------------
SOKNL: A novel way of integrating K-nearest neighbours with adaptive random forest regression for data streams

(Sun 2022) proposes self optimizing k-nearest neighbours (SOKNL) for data stream mining.

Related: 
    * See :ref:`2023-02-16 - FASLIP<2023-02-16 - FASLIP>` where author gave talk on this paper.
    * Another author discussed this paper on :ref:`2024-02-13 - ECRG <2024-02-13 - ECRG>`.

szegedy2013intriguing
---------------------
Intriguing properties of neural networks.

Notes: 
    * Adversarial attacks on neural networks. 
    * Trick neural nets into making the wrong prediction on purpose. 
    * Long tail problem of AI. 

TODO: 
    * https://arxiv.org/abs/1312.6199

szegedy2015going
----------------
Going deeper with convolutions

(Szegedy 2015) propose GoogLeNet, a CNN with inception modules.

tran2019genetic
---------------
Genetic programming for multiple-feature construction on high-dimensional classification.

(Tran 2019) propose multiple multi-tree GP methods for multi-class classification problems, including multi class-indepdent feature construction (MCIFC).

Notes:
    * Genetic programming for multiple-feature construction on high-dimensional Classification Data 
    * This paper includes an example of Multi-tree GP. 
    * I have apply Multi-tree GP for a one-vs-all multi-class classification problem. 

Related: 
    * See proposal for preliminary work section, MCIFC is used. 

tegmark2020aifeynman
--------------------
AI Feynman: A physics-inspired method for symbolic regression
    
Notes: 
    * Tegmark et al. developed they AI Feynman \cite{udrescu2020ai}. 
    * This algorithm can derive physics equations from data using symbolic regression. 
    * Symbolic regression is a difficult task, but by simplifying properties exhibited by physics equations (i.e symmetry, composability, separability), the problem can be reduced. 
    * Their work uses blackbox neural networks, to derive interpretable models that can easily be verified by humans. 

Related: 
    * See :ref:`(Tegmark 2022) <tegmark2020aifeynman2>` for the second iteration. 
    * Banzahf discussed the Feynman AI benchmark dataset at 2022-10-28 - ECRG. 
    * He employed correlation + linear scaling to exploit the shape of the data, a global measure, to find the best fit and reduce the search space. 

tegmark2020aifeynman2
-----------------------
AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity
    
Notes: 
    * 2nd iteration for the AI Feynman 2.0. 
    * More robust towards noise and bad data. 
    * Can discover more formulas that previous method. 
    * Implements Normalizaing flows. 
    * Method for generalized symmetries (abitrary modularity in the compuational graph formula)

Related: 
    * See :ref:`(Tegmark 2020) <tegmark2020aifeynman>` for the original AI Feynman. 
    * Banzahf discussed the Feynman AI benchmark dataset at 2022-10-28 - ECRG. 
    * He employed correlation + linear scaling to exploit the shape of the data, a global measure, to find the best fit and reduce the search space. 

tegmark2021aipoincare
---------------------
AI Poincaré 2.0: Machine Learning Conservation Laws from Differential Equations
    
(Temark 2021) use deep learning to model conservation laws from physics.

Notes: 
    * TODO [ ] READ 

tegmark2022poisson
------------------
Poisson Flow Generative Models

(Tegmark 2022) propose Poisson Flow Generative Models (PFGM)< which map a uniform distribution on a high-diemsnaioal hemisphere into any data distriubtion. 

Notes: 
    * TODO [ ] READ 

Related: 
    * See :ref:`2022-10-26 - Deep Learning<2022-10-26 - Deep Learning>`

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

van2008visualizing
------------------
Visualizing data using t-SNE.

(Van 2008) prose t-distributed stochastic neighbor embedding (t-SNE) for visualizing high-dimensional data.

Method: 
    * The t-SNE algorithm comprises two main stages. 
    1. First, t-SNE constructs a probability distribution over pairs of high-dimensional objects in such a way that similar objects are assigned a higher probability while dissimilar points are assigned a lower probability. 
    2. Second, t-SNE defines a similar probability distribution over the points in the low-dimensional map, and it minimizes the Kullback-Leibler divergence (KL divergence) between the two distributions with respect to the locations of the points in the map.

Related:
    * See :ref:`(Goodfellow 2016) <goodfellow2016deep>` chapter 3 pg. 72 for a derivation of Kullback-Leibler divergence. 

vaswani2017attention
--------------------
Attention is all you need

vincent2011connection
---------------------
A connection between score matching and denoising autoencoders

(Vincent 2011) propose a connection between score matching and denoising autoencoders.

Notes: 
    * Denoising score matcher 
    * Shows a simple denoising autoencoder training criterion is equivalent to matvching the score (with respect to the data) of a specific energy-based model to that of a nonparametric Pazen density estimator of the data. 

Related: 
    * See :ref:`(Goodfellow 2016) <goodfellow2016deep>` chapter 16, page 567, [Available] https://www.deeplearningbook.org/contents/graphical_models.html
    * See :ref:`2023-05-03 - Deep Learning <2023-05-03 - Deep Learning>`

volz2018evolving
----------------
Evolving mario levels in the latent space of a deep convolutional generative adversarial network

Related: 
    * :ref:`(Perez 2019) <perez2019analysis>` same author uses RHEA to design Game AI for ponnerman. 
    * :ref:`(Goodman 2020) <goodman2020weighting>` same user uses NBTEA to choose hyperparameters for balancing gamemplay.
    

von1986decision
---------------
"Decision trees". Decision Analysis and Behavioral Research.

(Von 1986) is cited on wikipedia for decision trees.

Available: https://cir.nii.ac.jp/crid/1130000795953711872

notes: 
    * see pp. 63-89, an excerpt from the book, Decision Analysis and Behavioural Research, 
    * decision trees are an algorithm that only contains conditional control statements, i.e. if-else statements.

Related: 
    * :ref:`(Breiman 1984) <breiman2017classification>` proposed the original CART algorithm for decision trees.

wang2018evolving
----------------
Evolving deep convolutional neural networks by variable-length particle swarm optimization for image classification

(Wang 2018) propose EvoCNN to automatically search for optimal CNN architecture without any manual work involved.

Related:
    * See :ref:`2022-10-27 - FASLIP<2022-10-27 - FASLIP>`

wang2020linformer
-----------------
Linformer: Self-attention with linear complexity},

`(Wang 2020) <https://arxiv.org/abs/2006.04768>` propose Linformer an :math:`O(n)` appromimation of self-attention.

Notes: 
    * Self-attention mechanism can be approximated with a low rank matrix. 
    * Reduces space and time complexity from :math:`O(n^2)` to :math:`O(n)`.
    * Performance on par with standard transformer models, whilst being much more memory and time efficient. 

Related:
    * See :ref:`(Peng 2023) <peng2023rwkv>` is inspired by Linformer.
    * See :ref:`(Zhai 2021) <zhai2021attention>` for Attention Free Transformer (AFT).
    * See :ref:`(Wang 2020) <wang2020linformer>` for Linformer paper.
    * See :ref:`(Kitaev 2020) <kitaev2020reformer>` for Reformer paper. 
    * See :ref:`(Katharopoulos 2020) <katharopoulos2020transformers>` for linear transformers. 
    * See :ref:`(Vaswani 2017) <vaswani2017attention>` for transformer paper.

watkins1992q
------------
Q-learning

(Watkins 1992) proposed q-learning, the foundation of reinforcment learning.

Related :
    * See 2022-12-05 - AJCAI #01

wayne2018unsupervised
---------------------
Unsupervised Predictive Memory in a Goal-Directed Agent

(Wayne 2018) propose the Memory, RL, and Inference Network (MERLIN), in which memory formation is guided by a process of predictive modeling.

Related:
    * See 2022-12-05 - AJCAI #01 

weinstein2022hunter 
-------------------
A Hunter Gatherer's Guide to the 21st Century (Book).

Notes:
    * pg. 229 "Evolutionary stable strategy - A strategy incapable of invasion by competitors"

Related: 
    * Steady-state algorithm mentioned in :ref:`2022-08-06 - ECRG <2022-08-06 - ECRG>`

white2023neural
---------------
Neural Architecture Search: Insights from 1000 Papers

(White 2023) is a literature survey of NAS.

Available: https://arxiv.org/abs/2301.08727

Related: 
    * See :ref:`(Zoph 2016) <zoph2016neural>` paper that popularized NAS.

wood2022automated
-----------------
Automated Fish Classification Using Unprocessed Fatty Acid Chromatographic Data: A Machine Learning Approach

Available: https://woodrock.github.io/#/AJCAI

Related: 
    * :ref:`(Black 2017) <black2017real>` used REIMS for fish fraud detection. 
    * :ref:`()`


wolpert1997no
-------------
    * No free lunch theorum. 
    * No classification algorithm that beats the rest for every problem. 
    * As training instances approaches infinity, classification accuracy on all distributions of noise, approaches predicting mean class. 
    * All machine learning algorithms are task specific, don't generalize to all problems, no artifical general intelligence (AGI), yet... 

xie2022physics
--------------
A Physics-Guided Reversible Residual Neural Network Model: Applied to Build Forward and Inverse Models for Turntable Servo System

Related: 
    * Author gave talk in :ref:`2023-05-25 - FASLIP <2023-05-25 - FASLIP>`
    * Residual Neural Networks - Resnet, see :ref:`(He 2016) <he2016deep>`

xiong2020layer
--------------
On layer normalization in the transformer architecture

(Xiong 2020) propose the Pre-Layer Normalization (Pre-LN) for transformers.

Available: https://proceedings.mlr.press/v119/xiong20b

Related: 
    * Transformers :ref:`(Vaswani 2017) <vaswani2017attention>`

xin2022current
--------------
    * Do Current Multi-Task Optimization Methods in Deep Learning Even Help?
    * A paper that is strongly against mutli-task learning. 
    * TODO [ ] READ

xue2014particle
---------------
    * Brown et al. proposed a PSO with novel initialising and updating mechanisms.
    * The initialization strategy utilized both forward and backwards selection.
    * The updating mechanism overcame the limitations of the traditional method by considering the number of features.
    * The proposed algorithm had better performance in terms of computing, fewer features selected and classification accuracy.

xue2015survey
-------------
A survey on evolutionary computation approaches to feature selection

(Xue 2015) is a literature survey of EC methods for feature selection 

Related: 
    * Ruwang quoted this paper in :ref:`2023-12-08 - ECRG <2023-12-08 - ECRG>`


yang2022noise
-------------
Noise-Aware Sparse Gaussian Processes and Application to Reliable Industrial Machinery Health Monitoring

(Yang 2022) proposed a Noise-Aware Sparse Gaussain Process (NASGP) with Bayesian Inference Network. 

Available: https://ieeexplore.ieee.org/abstract/document/9864068/

Data: 
    * Domain - maintainace of machinary equipment requires real-time health monitoring. Most state-of-the-art models require high quality monitoring data, but are not robust to noise present in real-world applications. 
    * Problem - predict an estimate of the reamining useful life of machinary equipment using noisy data. 

Method: 
    * Noise-Awate Sparse Gaussain Processes (NASGP) + Bayesian Inference Network. 

Results: 
    * NASGP are capable of high-performance and credible assessment under strong noises. 
    * Developed a generative additive model to bridge the gap between latent inference mechanism and domain expert knowledge. 
    * Method worked well in two different domains: (1) remaining useful life prognosis, (2) fault diagnosis in rolling bearings. 

Why it matters?
    * The method is robust to noise, and can be applied to real-world applications, not just academic benchmarks (toy datasets). 
    * Method provides a generative additive model that works well in two different domains.
    * Important to monitor machinary equipment in real-world applications, to ensure safety, automation, and efficiency.

Related: 
    * See :ref:`2022-10-12 - Deep Learning<2022-10-12 - Deep Learning>` for more 

yu2019adapting
--------------
Adapting BERT for target-oriented multimodal sentiment classification

(Yu 2019) propose TomBERT for multi-modal sentiment classification.

Available: https://ink.library.smu.edu.sg/sis_research/4441/

Related: 
    * Discussed in :ref:`2024-02-29 - FASLIP <2024-02-29 - FASLIP>`

zemmal2016adaptative
--------------------
Adaptative S3VM Semi Supervised Learning with Features Cooperation for Breast Cancer Classification 

(Zemmal 2016) propose S3VM, a semi-supverised SVM, that combines labelled and unlablled datasets, to improve SVM performance for Breast Cancer Classification. 

Data: 
    * Domain - breast cancer is the most common cause of cancer and the second leading cause of all cancer deaths. Early detection in the intiial stages of cancer is crucial for survival.
    * Problem - Breast Cancer Classification with computer-aided diagnosis (CAD), to classify breast cancer tumours as malignant or benign. 
    * Collecting data is expensive and time-consuming, practitioners wait 5 years after treatment to label a patient as survived or not. Label annotation of the datasets is a slow processs, but their is an abundance of unlabelled data. 

Method: 
    * Supervised + unsupervised learning to boost SVM performance. 
    * Using unlabeleld data (unsupervised) to ensure the decision boundaries are drawn through low density areas. 
Results: 
    * Evaluate method by increasing the proportion of labelled-to-unlabelled data for each test (rarely, moderately low, moderate). 
    * Promising results were validated on a real-world dataset of 200 images. 

Why it matters? 
    * SVM performance can be improved by using unlabelled data.
    * Unlabelled data is abundant, but expensive to label. Methods that utilize unlabelled data are cheap, efficient, and improve performance. 
    * Early cancer detection is crucial for survival. 

Related: 
    * TODO [ ] - read. 

zhai2021attention
-----------------
An attention free transformer

`(Zhai 2021) <https://arxiv.org/abs/2105.14103>`__ is an apple paper that presents the Attenion Free Transformer (AFT).

Notes: 
    * Recurent Neural Network for inference. 
    * Cheaper inference method for GPT-like transformer models. 

Related: 
    * See :ref:`2023-06-10 - Deep Learning <2023-07-10 - Deep Learning>` where paper was dicussed. 
    * See :ref:`(Peng 2023) <peng2023rwkv>` for RWKV paper that builds on this. 
    * See :ref:`(Wang 2020) <wang2020linformer>` for Linformer paper.
    * See :ref:`(Kitaev 2020) <kitaev2020reformer>` for Reformer paper. 
    * See :ref:`(Katharopoulos 2020) <katharopoulos2020transformers>` for linear transformers. 
    * See :ref:`(Vaswani 2017) <vaswani2017attention>` for transformer paper. 

zhang2008two
------------
    * Zhang et al. proposed a 2-D COW algorithm for aligning gas chromatography and mass spectrometry. 
    * The algorithm warps local regions of the data to maximise the correlation with known reference samples. 
    * This work uses data fusion with labelled reference samples, to improve the quality of new samples.

zhang2021evolutionary
---------------------
    * An Evolutionary Forest for Regression 
    * Hengzhe Zhang's paper from ECRG. 
    * TODO [ ] READ

zhang2023adding
---------------
Adding conditional control to text-to-image diffusion models

Methods: 
    * Augment existing LDM, e.g. StableDiffusion, with ControlNets, 
    * to enable conditional inputs like: 
        1. edge maps, 
        2. segmentation maps, 
        3. and keypoints.

Results:    
    * See figure 1, https://twitter.com/jrhwood/status/1653612277294309378?s=20
    * Input: the cany edge map (bottom left)
    * Prompt: "A high-quality, detailed, professional image"
    * Output: the 4x generated images on the right

Applications: 
    * Image generation 
    * Fine-grained control over LDM output 
    * Memes in art styles https://twitter.com/jrhwood/status/1653612282935656449?s=20

Related: 
    * See :ref:`2023-05-03 - Deep Learning <2023-05-03 - Deep Learning>`
    * See :ref:`(Ho 2020) <ho2020denoising>` for more on diffusion models.
    * See :ref:`(Song 2020) <song2020denoising>` for more on diffusion models.

zhao2019maximum
---------------
Maximum relevance and minimum redundancy feature selection methods for a marketing machine learning platform. 

(Zhao 2019) propose a feature selection method for a marketing machine learning platform.

Intro: 
    * This (Zhao 2019) is a paper from Uber engineering. 
    * Business objectives: (1) user acquisition, (2) cross/up sell, (3) user churn. 
    * Curse of dimensionality: ineffeciency, overfitting, high maintance, low intrepretability. 
    * FS enabled beter compliance/troubleshooting, business intiution and insights. 
    * Smaller problem space for troubleshooting and diagnosis. 
    * By only using important features for prediction task, it is easier to interpret what features/patterns the model is using. 
    * The m best features are not the best m features - many features are correlated and redundant. 
    * MRMR is a filter bases FS method that considers both: (1) relevance for predicting outcome, (2) redundancy within selected features. 

Background:
    * Mutual Information (MI): 
        * is a measure of the mutual depedence between two random variables. 
        * :math:`I(X;Y) = H(X) - H(X|Y)`, the amount of information one can geain about one random variable from another. 
        * :math:`I(X;Y) = D_{KL}(P_{(X,Y)} || P_X \otimes P_X)`, let :math:`(X,Y)` be a pair of random variables, take the KL divergence between their join distribution :math:`P_{(X,Y)}` and the product of their maginal distribution :math:`P_X \otimes P_X`.
    * MRMR
        * For the MRMR framework, the feature importance can be expressed as :math:`f^{mRMR} = I(Y,X_i) - \frac{1}{|S|} \sum_{X_s \in S} I(X_s;X_i)`. where
            - :math:`S` is the set of selected features. 
            - :math:`|S|` ois the size of the feature set.
            - :math:`X_s \in S` is one features of the set :math:`S`
            - :math:`X_i` denotes a feature is currently not selected. 
            - The function :math:`I(.;.)` os the mutual information.   
        * It builds a set of best features based of maximum feature importance each iteration.

Datasets: 
    * 3x real-world, 1x synthetic. 
    * Goal: robust FS method that generalizes to many datasets. 
    
Method: 
    * Extensions are based on relatedness to downsteam machine learning models those features are then used on. 
    * RDC can identify redundancy in non-linear relationships. 
    * Random-Forest correlation quotient (RFCQ) uses the feature importance metric from random forest.
    * Issues: scale differences  between relevance and redundancy metrics. 
    * Metrics: computational efficiency (speed) and classification accuracy.
    * The FS methods (8) x classifiers (3) x datasets (4) are all combined to produce a multiplicity (96) sets of results. 
    * Splines used to generated various kinds of features for the synthetic dataset. 
    * Computation efficiency (speed) is a useful metric for motivating FS methods. 
    * Correlation heatmaps are an effectieve way to visualize correlation and redundancy in a dataset. Motives FS methods. 
    * Box and whisker plots provide a stunning visual for comparison of classification performance across different FS methods. 
    * Metadata is provided for each dataset, i.e. Number of features, Number of users. 
    * Random forest classifier is run twice using different parameters, explicit sklearn parameters for python given for reproduceability. 
    
Why it matters? 
    * Could include "Implementation in Production" section in my thesis, even if theoretical, to ground work in real-world application. 
    * Future work/alterantive approaches are discussed in conclusion, they propose additional extenions of MRMR. 
    * Nice to give back to the research community by thanking reviewers in the acknowledgements. 

Related: 
    * MRMR :ref:`(Ding 2005) <ding2005minimum>` uses mutual information to measure both relevance and redundancy.
    * Mutual information can be given for a discrete and continuos by a double sum and integral respectively. See :ref:`(Goodfellow 2016) <goodfellow2016deep>` chapter 3 pg. 72 for a derivation of Kullback-Leibler divergence. 
    * :ref:`(Brown 2012) <brown2012conditional>` generalizes information based FS methods, e.g. MRMR, into conditional likelihood framework.
    * Two FS papers, (:ref:`Lui 1995 <liu1995chi2>`, :ref:`Zhao 2019 <zhang2008two>`) use a synthetic datasets where redundant features are known.

zhu2022few
----------
A few-shot meta-learning based siamese neural network using entropy features for ransomware classification},

Related: 
    * :ref:`(Bromley 1993) <bromley1993signature>` is the original siamese network paper.
    * :ref:`(Jing 2022) <jing2022masked>` proposed masked siamse convnets for few-shot learning.

zoph2016neural
--------------
Neural Architecture Search: Insights from 1000 Papers

Available: https://arxiv.org/abs/2301.08727

Related: 
    * Mentioned in :ref:`2022-09-21 - FASLIP <2023-09-21 - FASLIP>`
    * See :ref:`(White 2023) <white2023neural>` for a literature survey of NAS.
