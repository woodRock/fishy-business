Literature Review
=================

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

bao2022estimating
-----------------
    * Estimating the Optimal Covariance with Imperfect Mean in Diffusion Probabilistic Models 
    * Diffusion Probabilistic Models (DPM) are special Markov Models with Gaussian Transitions. 
    * Paper shows how to go from noisy-to-clean with a deterministic process. 
    * A new approach to diffusion based models.

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

brewer2006brown
---------------
    * Flashbuld memories - recollections that seem vivid and clear, so we take them to be accurate. 
    * Most likely occur for distinct stronly positive or negative emotional events. 
    * Weddings, Funerals, Deaths, Tragedy, Violence. 
    * We are more likely to be confident these are correct.
    * But our memory is shit, so we often re-write and incorrectly recall these events. 
    * The distinictness of flashbulb memories, does help recall them longer, but does not guarantee correctness. 

brochu2010tutorial
------------------
    * A Tutorial on Bayesian Optimization of Expensive Cost Functions
    * Application: 
        1. Active User Modeling 
        2. Hierarchical Reinforcement Learning
    * Covers the theory and intuition behind Bayesian optimizaiton with visual examples. 
    * Discusses preference galleries, hierachichal control
    * Recommended reading from the FASLIP talk on Bayesian Optimizatio 2022-03-24.
    * **TODO** read this! 

brown2012conditional
--------------------
    * Conditional likelihood maximisation: a unifying framework for information theoretic feature selection
    * Generalized model for information based feature selection methods. 
    * These models generazlize to iterative maximizers of conditional likelihood. 

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

chen2019looks
-------------
    * This looks like that: deep learning for interpretable image recognition
    * Add a prototype layer to neural networks to for interpretable models for black-box nets. 

chen2021evaluating
------------------
    * 70% accuracy for basic DSA problems. 
    * Can't solve more difficult problems - doesn't optimize solutions for performance. 
    * CoPilot outperforms other state-of-the-art NLP code generation models. 
    * Requires "fine-tuning", supervised human intervention to hint towards correct answer. 

codevilla2018end 
----------------
    * High-speed autonomous drifting with deep reinforcement learning. 
    * Far easier to use real-world data on driving that has already been collected than generate simulation data. 
    * Data augmentation used to help network generalize to new scenarios and edge cases not in the training data. 
    
cortes1995support
-----------------
    * Cortes and Vapnik proposed the Support Vector Machine (SVM).
    * This model creates a hyperplane that can draw distinct class boundaries between classes.
    * We call these class boundaries the support vectors.
    * We are performing multi-class classification, so it used a one-vs-all approach \cite{sklearn2021feature}.
    * This creates a divide between one class and the rest, then repeats for the other classes.

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

ding2005minimum
---------------
    * Minimum Redundancy - Maximum Relevance (MRMR)


eder1995gas
-----------
    * Gas chromatography (GC) \cite{eder1995gas} is a method that can identify chemicial structures in these fish oils.
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

eyesenck1980effects
-------------------
    * Effects of processing depth, distinctiveness, and word frequency on retention. 
    * In general distinct stimuli are better remembered than non-distinct ones. 
    * We are more likely to remember things that are out of the blue, or that have a personal connection to us. 

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

godden1975context
-----------------
    * Context-dependent memory in two natural environments: On land and underwater. 
    * Scuba divers who learn lists of words underwater, best recalled them underwater. 
    * Same true for words learnt on land. 
    * Recall accuracy depends on similarity of context in sensory information. 

hand2001idiot
-------------
    * Naive bayes. 

ho1995random
-------------
    * Random forest. 

Hofstadter1979godel 
-------------------
    * Godel Escher Bach 
    * The hand that draws itself. 

jing2020learning
----------------
    * Graph nerual Networks can be used for protien folding. 
    * Equivariance to rotations - if the networks thinks the same instance rotates is a completely different structure, this is very inefficient. 
    * Instead we want rotation invariant representations for things like protiens. (Like we wan't time invariant representations for gas chromatography). 
    * Voxels are 3D pixels, these can be used to make a 3D representation of an instance, which then applies a 3D Convolutional Neural Network. 
    * We think that (1) message passing and (2) spatial convolution, are both well suited for different types of reasoning. 
    * In protein folding, their are chemical propoerties of protiens that simplify the combinatorial search space for the graphical neural network. 
    * This is similar to how the AI Feynman (Tegmark 2020) used properties of physics equations to simplify symbolic regression. 

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

kennedy1995particle
-------------------
    * Original PSO algorithm.

kennedy1997discrete
-------------------
    * PSO for feature selection. 
  
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

kononenko1994estimating
-----------------------
    * Estimating attributes: Analysis and extensions of Relief. 
    * ReliefF paper, an extension of Relief (Kira 1992, kira1992practical)
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

lehman2020surprising
--------------------
    * The surprising creativity of digital evolution: A collection of anecdotes from the evolutionary computation and artificial life research communities
    * Annecdotes from researchs in EC about their algorithms demonstrating bizzare interesting behaviour. 

liu1995chi2
-----------
    * Chi2: feature selection and discretization of numeric attributes
    * Discretization bins continuous values into discrete ones.  
    * Feature selection via discretization - ideal for numeric data.
    * Motivation: (1) (can) improve performance, (2) efficiency (time/space), (3) simplify models. 
    * Chi2 discretizes and performs FS - useful as many algorithms perform better with discrete/binary data. 
    * Under discretization would return the original continuous attribute unchanged. 
    * Over-discretization is when inconsistencies are introduced to the data - the data loses fidelity. 
    * Previous work, ChiMerge, with hyper-parameter :math:`\alpha` the significance level that had to be manually set. 
    * :math:`\alpha` is nuisance variable that requires black magic approach to tune.
    * Difficult to find ideal :math:`\alpha` without domain knowledge or extensive trial and error. 
    * New approach Chi2 lets data determine value of :math:`\alpha`, perform discretization until over-discretization - a stopping criterion. 
    * Chi2 is a two-phase method, a generalized version of ChiMege that automatically determines a good :math:`\chi^2` threshold that fits the data.
    * The formula for calcutaling the $\chi^2$ statistic is given by, :math:`\chi^2 = \sum_{i=1}^2 \sum_{j=1}^k \frac{(A_{ij} - E_{ij})^2}{E_{ij}}`.
    * Phase 1: Extends ChiMerge to be an automated one, to select an ideal value for :math:`\alpha` based on the data. 
    * Phase 2: Each feature is assigned signfnicance level and merged in a round robin fashion - until stopping criterion met. 
    * Attributes only merged to one value are elminianted as part of feature selection. 
    * Degrees of freedom: the maximum number of logically independent values, which are values that have the freedom to vary, :math:`D_F = N - 1`, where :math:`N =` samples, :math:`D_F =` degrees of freedom. 
    * If :math:`R_i` or :math:`C_i` is zero, set to 0.1. Similar to zero frequency problem from Naive Bayes. I.e. Multiplication by zero is always 0, so all other information is lost. 
    * Experiments: DT (C4.5), Data with Noise, and Synthetic data. 
    * Datasets: Iris (continious), Breat (discrete), Heart (mixed).
    * C4.5, a DT classification algorithm, is run on its default setting.
    * Results show predictive accuracy and size, same or improved for all datasets where Chi2 was applied.
    * Chi2 was able to remove noise (irrelvant features) from synthetic and real world data.
    

loh2011classification
---------------------
    * Decision tree. 

mantyla1998cue
--------------
    * Cue distinctiveness and forgetting: Effectiveness of self-generated retrieval cues in delayed recall. 
    * Students were given a word list, and asked to make 1 or 3 retrieval cues. 
    * Students with who used their own multiple retrieval cues had better recall.
    * Recall was terrible when using another students own personal retrieval cues. 
    * Multiple self-generated retrieval cues is the most effective approach to maximising recall. 

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

moraglio2012geometric
---------------------
    * Genetic semantic genetic programming. 
    * **TODO** read - related to Qi Chen talk on 2022-03-18 ECRG. 
    * Unimodal fitness landscape, one global optima, but semantic search is intractable. 
    * We approximate semantic search through geometric genetic programming methods. 


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

pearce2021empirical
-------------------
    * 70% accuracy for basic DSA problems. 
    * Can't solve more difficult problems - doesn't optimize solutions for performance. 
    * CoPilot outperforms other state-of-the-art NLP code generation models. 
    * Requires "fine-tuning", supervised human intervention to hint towards correct answer. 

qin2021one
----------
    * From One Hand to Multiple Hands: Imitation Learning for Dexterous Manipulation from Single-Camera Teleoperation. 
    * TODO read 
    * This paper shows single-camera teleoperation capabilities for SCARA. 
    * This could be used to allow for remote intervention in edge cases for our SCARA. 

raine1997brain
--------------
    * Muderers pleading not guilty be reason of insanity (NGRI).
    * Pre-disposition to less activity in their pre-frontal cortex. 
    * Pre-frontal cortex associated with goal-directed planning and delayed gratification. 
    * Different brain chemistry meant more likely to perform violent impulsive behaviour. 
    * Justification for lebotomy - electrocution of pre-frontal cortex - now replaced by anti-psychotics. 

ramesh2022hierarchical
----------------------
    * Hierarchical Text-Conditional Image Generation with CLIP Latents. 
    * a.k.a. Dalle 2 and Very popular on the internet.
    * Original was a d-VAE (discrete), Dalle 2 is a diffusion based model that uses CLIP. 
    * CLIP trains an auto-enocder to have minimize the distance between image and text embeddings in the latent space. 
    * Those image embeddings are fed to an autoregressive or diffusion prior to generate image embeddings. 
    * Then this embedding is used to condition a diffusion decoder which produces an image. 
    * The model is trained on 250 Million images, and has 3.5 billion parameters. 
    * We can use CLIP to interpolate between two images in the latent space. 
    * As we increase the dimensionality of the latent space we can represent more complex hierarchical structures. 
    * CLIP fails at producing text, and reconstruction can mix up objects and their attributes. 

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

shahriari2015taking
-------------------
    * Taking the Human Out of theLoop: A Review of Bayesian Optimization.
    * Recommended reading from the FASLIP talk on Bayesian Optimizatio 2022-03-24.
    * **TODO** read this. 

song2020denoising
-----------------
    * Denoising diffusion implicit models. 
    * See 2022-07-06 - Deep Learning

szegedy2013intriguing
---------------------
    * Intriguing properties of neural networks.
    * Adversarial attacks on neural networks. 
    * Trick neural nets into making the wrong prediction on purpose. 
    * Long tail problem of AI. 

tegmark2020ai
-------------
    * Tegmark et al. developed they AI Feynman \cite{udrescu2020ai}. 
    * This algorithm can derive physics equations from data using symbolic regression. 
    * Symbolic regression is a difficult task, but by simplifying properties exhibited by physics equations (i.e symmetry, composability, separability), the problem can be reduced. 
    * Their work uses blackbox neural networks, to derive interpretable models that can easily be verified by humans. 

tegmark2020ai2
--------------
    * 2nd iteration for the AI Feynman 2.0. 
    * More robust towards noise and bad data. 
    * Can discover more formulas that previous method. 
    * Implements Normalizaing flows. 
    * Method for generalized symmetries (abitrary modularity in the compuational graph formula)

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

zemmal2016adaptative
--------------------
    * S3VM - semi-supverised SVM. 
    * Using unlabeleld data to ensure the decision boundaries are drawn through low density areas. 
    * TODO - read. 

zhang2008two
------------
    * Zhang et al. proposed a 2-D COW algorithm for aligning gas chromatography and mass spectrometry. 
    * The algorithm warps local regions of the data to maximise the correlation with known reference samples. 
    * This work uses data fusion with labelled reference samples, to improve the quality of new samples.

zhao2019maximum
---------------
    * Maximum relevance and minimum redundancy feature selection methods for a marketing machine learning platform. 
    * A paper from Uber. 
    * Business objectives: (1) user acquisition, (2) cross/up sell, (3) user churn. 
    * Curse of dimensionality: ineffeciency, overfitting, high maintance, low intrepretability. 
    * FS enabled beter compliance/troubleshooting, business intiution and insights. 
    * Smaller problem space for troubleshooting and diagnosis. 
    * By only using important features for prediction task, it is easier to interpret what features/patterns the model is using. 
    * The m best features are not the best m features - many features are correlated and redundant. 
    * MRMR is a filter bases FS method that considers both: (1) relevance for predicting outcome, (2) redundancy within selected features. 
    * Original MRMR uses mutual information to measure both relevance and redundancy. 
    * Information based FS methods were generalized into a conditional likelihood framework (Brown 2012, brown2012conditional).
    * Mutual Information (MI): is a measure of the mutual depedence between two random variables. 
    * :math:`I(X;Y) = H(X) - H(X|Y)`, the amount of information one can geain about one random variable from another. 
    * :math:`I(X;Y) = D_{KL}(P_{(X,Y)} || P_X \otimes P_X)`, let :math:`(X,Y)` be a pair of random variables, take the KL divergence between their join distribution :math:`P_{(X,Y)}` and the product of their maginal distribution :math:`P_X \otimes P_X`.
    * For the MRMR framework, the feature importance can be expressed as :math:`f^{mRMR} = I(Y,X_i) - \frac{1}{|S|} \sum_{X_s \in S} I(X_s;X_i)`. where
        - :math:`S` is the set of selected features. 
        - :math:`|S|` ois the size of the feature set.
        - :math:`X_s \in S` is one features of the set :math:`S`
        - :math:`X_i` denotes a feature is currently not selected. 
        - The function :math:`I(.;.)` os the mutual information. 
    * Mutual information can be given for a discrete and continuos by a double sum and integral respectively. See (Goodfellow 2016, goodfellow2016deep) for a derivation of Kullback-Leibler divergence. 
    * It builds a set of best features based of maximum feature importance each iteration. 
    * Extensions are based on relatedness to downsteam machine learning models those features are then used on. 
    * RDC can identify redundancy in non-linear relationships. 
    * Random-Forest correlation quotient (RFCQ) uses the feature importance metric from random forest.
    * Issues: scale differences  between relevance and redundancy metrics. 
    * Datasets: 3x real-world, 1x synthetic. 
    * Goal: robust FS method that generalizes to many datasets. 
    * Metrics: computational efficiency (speed) and classification accuracy.
    * The FS methods (8) x classifiers (3) x datasets (4) are all combined to produce a multiplicity (96) sets of results. 
    * Both (Lui 1995, liu1995chi2), and this - two FS papers, use a synthetic dataset where redundant features are known. 
    * Splines used to generated various kinds of features for the synthetic dataset. 
    * Computation efficiency (speed) is a useful metric for motivating FS methods. 
    * Correlation heatmaps are an effectieve way to visualize correlation and redundancy in a dataset. Motives FS methods. 
    * Box and whisker plots provide a stunning visual for comparison of classification performance across different FS methods. 
    * Metadata is provided for each dataset, i.e. Number of features, Number of users. 
    * Random forest classifier is run twice using different parameters, explicit sklearn parameters for python given for reproduceability. 
    * Could include "Implementation in Production" section in my thesis, even if theoretical, to ground work in real-world application. 
    * Future work/lterantive approaches are discussed in conclusion, they propose additional extenions of MRMR. 
    * Nice to give back to the research community by thanking reviewers in the acknowledgements. 
