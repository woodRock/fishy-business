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

banzhaf2009genetic
------------------
    * Genetic Programming: An Introduction On The Automatic Evolution Of Computer Programs And Its Applications
    * TODO [ ] must read book for foundations of GP. (buy?)

bengio2017consciousness
-----------------------
    * The consciousness prior

bi2020gc
--------
    * Bi et al. proposed a CNN model that incorporated GC-MS data fusion for food science.
    * The high-dimensional data was naturally suited towards the CNN.
    * Their work classified the flavour quality of peanut oil with 93\% accuracy.
    * Similar to this project, the existing technique for analysis was intractable large scale.
    * The fusion of existing datasets improved the efficacy of their model.

black2019rapid
--------------
    * Rapid detection and specific identification of offals within minced beef samples utilising ambient mass spectrometry
    * Rapid evaporative ionisation mass spectrometry (REIMS) is one example of this, and has been used to detect horse meat contamination in beef.

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

brudigam2021gaussian
--------------------
    * Gaussian Process-based Stochastic Model Predictive Control for Overtaking in Autonomous Racing
    * See 2022-07-20 - Deep Learning 

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
    * A deep learning method for bearing fault diagnosis based on cyclic spectral coherence and convolutional neural networks
    * (Chen 2022) propose a Cyclic Spectral Coherence (CsCoh) + Convolutional Neural Networks (CNNs) for rolling element fault diagnosis. 
    * Data: 
        * The domain is rolling element fault diagnosis - i.e. ball bearings in a factory setting. 
        * A rotating bearing will modulate (go up and down) in ptich in a non-periodic manner, this is a telltale sign of a faulty ball bearing. 
    * Method: 
        * Combine CsCoh + CNNs for fault diagnosis of rotating elements in a factory. 
        * Cyclic Speherical Coherence (CsCoh) is used to preprocess virbation signals, estimated by the fourier transform of Cyclic ACF (see paper for derivation). 
        * Group Normalization (GN) is developed to reduce the internal covariant shift by data distribution discrepency, extends applications of the algorithm to real industrial environments. 
    * Results: 
        * Their proposed method improves classification performance, >95% accuracy needed for use in real-world. 
        * CsCoh proivde superior dsciminate feature representations for bearing health statuses under varying conditions. 
        * Group Normalization increases robustness for data from differenet domains (with different data distributions). 
    * Why it matters? 
        * Garbage-in-garbage out - Preprocessing can dramatically improve the performance of a CNN.
        * Group Normalization makes the method robust, and applicable to out-of-distribution data from unseen domains. 
        * Detecting faults in ball bearings is crucial for safety, automation, and efficiency in factories.
    * See 2022-10-12 - Deep Learning for more. 

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

chevalier2018babyai
-------------------
    * Babyai: A platform to study the sample efficiency of grounded language learning

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


ding2005minimum
---------------
    * Minimum Redudancy Featyre Selection from MicroArray Gene Expression Data. 
    * Original Minimum Redundancy - Maximum Relevance (MRMR) paper. 
    * See (Zhao 2019, zhao2019maximum) for more recent Uber paper.


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

riccardo2009field
-----------------
    * A Field Guide to Genetic Programming
    * A free resource for GP research available online. 


fawzi2022discovering 
--------------------
    * Discovering faster matrix multiplication algorithms with reinforcement learning 
    * Deep Mind - AlphaTensor 
    * Improves Strassman's algorithm for 4x4 matrix multiplication for first time in 50 years.
    * Matrix multiplication is the bedrock of deep learning. 
    * Fast matrix multplication can lead to exponential speedups in deep learning.
    * TODO [ ] - Read this paper 

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

girshick2014rich
----------------
    * Rich feature hierarchies for accurate object detection and semantic segmentation 
    * R-CNNs, Region-based Convolutional Neural Networks.
    * Combine region proposals and CNNs. 
    * See FASLIP - 2022-10-06 for more details.

godden1975context
-----------------
    * Context-dependent memory in two natural environments: On land and underwater. 
    * Scuba divers who learn lists of words underwater, best recalled them underwater. 
    * Same true for words learnt on land. 
    * Recall accuracy depends on similarity of context in sensory information. 

grcic2021densly
---------------
    * Normalizing flows are bijective mappings between input and latent representations with a fully factoritzed distribution. 
    * Normalizing flows (NF) are attrictive due to exact likelihood evaluation and efficient sampling. 
    * However their effective capacity is often insuffiencet since bijectivity constraints limit the model width. 
    * The proposed method addresses this limitation by incrementally padding intermediate representations with noise. Precondition noise in accordance with previous invertible units, coined "cross-unit coupling".
    * Their invertible glow0like, modules increase the expressivity by fusing a densely connected block with NYstron self-attention. 
    * They refer to their proposed achitecture as DenseFlwo, since both cross-unit and intra-module couplings rely on dense connectivity. 
    * Experiments show significant improvements due to prposed contributions and reveal state-of-the-art density estimation under moderate computing budgets. 

he2020bayesian
--------------
    * Bayesian deep ensembles via the neural tangent kernel

hand2001idiot
-------------
    * Naive bayes. 

hildebrandt2010towards
----------------------
    * Towards improved dispatching rules for complex shop floor scenarios: a genetic programming approach  


ho1995random
-------------
    * Random forest. 

Hofstadter1979godel 
-------------------
    * Godel Escher Bach 
    * The hand that draws itself. 

jacot2018neural
---------------
    * Neural tangent kernel: Convergence and generalization in neural networks

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

ke2018sparse
------------
    * Sparse attentive backtracking: Temporal credit assignment through reminding

kennedy1995particle
-------------------
    * Particle Swarm Optimisation (PSO). 
    * Purpose: POS optimizes non-linear functions with particle swarn methedology. 
    * Applications: (1) non-linear function optimization, (2) neural network training. 
    * PSO was discovered through simulation of a simpleified social behaviourmodel. Then taken from a social behaviour model, and turned into an optimizer. 
    * Model is very simple, requires a few lines of code, primitive mathematics operators, both effecient in memory and speed. 
    * Applications: Train ANN weights, Model Schaffers f6 function a GA from (Davis 1991). 
    * Paradigms: (1) Artificial life - i.e. fish schooling, birds flocking, (2) Genetic algorithms / evotionary programming. 
    * School of Fish https://youtu.be/15B8qN9dre4
    * (Reynolds 1987) was intrigued by the aesthetics of bird flocking, the choreography, synchonocity. He wanted to understand the mechanics of bird flocking - as set of simple rules that governed the behaviour. 
    * With the assumption, like Conway's Game of Life for cellular automata, that a simple set of rules, my underpin the unpredictable and complex group dynamics of bird social behaviour. 
    * The synchonicit was though of as a function of the bird trying to maintain an optimal distance between itself and its neighbours.
    * Perhaps these same rules govern social behaviour in humans. Social sharing of infomration amoung members of the same species (cospeciates) offers an evolutionary advantage (Wilson 1975).
    * Motivation for simulation: to model human behaviour. Humans are more complex, we don't just update our velocity/direction as animals flocking do, we update our beliefs/views to conform to our peers around us - i.e. social desirability bias, cultural homogenuity. 
    * In abstract multi-dimenisional space, our psychological space, we allow colluions within a population - i.e. two individuals may share the same beliefs. Thus our model allows collisions, e.g. "collision-proof birds". 
    * Aristotle spoke of Qualitative and quantitative movement. 
    * Initial approach: a nearest neighbour method to synchonocity that matched velocity resulted in unifrom unchanging direction. 
    * Stochasity, randomness, "craziness" was required to add variation to the flocks direciton. Enough stochacity to give the illusion of aritificial life. 
    * (Heppner 1990) had simulations which introduced a "roost", a global maximum, or home the birds, that they all know. 
    * But, how do birds find food? I.e. a new bird feeder is found within hours. 
    * Agents move towards their best know value - the cornfield, in search of food. 
    * Birds store their local maxima, the cornfield vector (I know there is food here!). 
    * All birds in the flock know the global best position, the roost. 
    * Simulation behaviour: a high p/g increment had violent fast behaviour, an approximately equal p/g increment had synchronocity, low p/g increment had no convergence.
    * Improvements: removed craziness, removed nearest neighbour (NN), without NN collisions were enabled, the flock was now a swarm. A swarm not a flock, because we have collisions. 
    * g/p increment values had to be chosen carefully. 
    * Social anaologies: :math:`pbest` is autiobiographical memory, :math:`\nabla pbest` is simple nostalgia. :math:`gbest` is public knowledge, :math:`\nabla gbest` is social conformity. 
    * Appxomiations, PSO could solve the XOR problem on a 2-3-1 ANN with 13 parameters. 
    * Improvement: velocities were adjusted according to their difference, per dimension, this added momementum, a memory of previous motion. p/g increment was a nuisance parameter, and was such removed. 
    * Stochastic factor, which amplifieid the randomness, was set to 2. This makes the agents "overfly" or overshoot the target about half of the time. Tuned with black magic, a more formal derivation could be done in future work. 
    * Tried a model with one midpoint between :math:`gbest` and pbest, but it converged at the midpoint. 
    * The stochasity was necesarry for good results. 
    * Explorers and settlers model, explorers overrun target, settlers more precise, had little improvement, Occam's razor removed the complex model. 
    * Version without momentum, had no knowledge of previous motion, and failed to find the global optima. 
    * (Millonas 1995) developed 5 basic principles of swarm intelligence. 
        1. Prxomity - perform space/time computations. 
        2. Quality - respond to quality features in the environment 
        3. Diversity - not commit to narrow channels. 
        4. Stablity - Don't change mode behaviour each iteration. 
        5. Adaptability - Change behaviour if it is worth it. 
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
    * PSO walks a fine line between order (known) and chaos (unknown). 
    * Philosophy (some beautiful philosophical musings from the end of the paper) 
        * Allows wisom to emerge rather than impose it. 
        * Emulates nature rather than trying to control it. 
        * Makes things simpler than more complex.

kennedy1997discrete
-------------------
    * PSO for feature selection. 

kerber1992chimerge
------------------
    * Chimerge: Discretization of numeric attributes 
    * Predecessor to Chi2 (Liu 1995, liu1995chi2)
  
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

lecun1998gradient
-----------------
    * Gradient-based learning applied to document recognition
    
lee2019wide
-----------
    * Wide neural networks of any depth evolve as linear models under gradient descent

lehman2020surprising
--------------------
    * The surprising creativity of digital evolution: A collection of anecdotes from the evolutionary computation and artificial life research communities
    * Annecdotes from researchs in EC about their algorithms demonstrating bizzare interesting behaviour. 

lin2017feature
--------------
    * Feature pyramid networks for object detection. 
    * Feature Pyramid Network (FPN)
    * See FASLIP - 2022-10-06 for more details.

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
    * Cybermarine research magazine aims. girshick2014rich
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

muller2021transformers
----------------------
    * Transformers Can Do Bayesian Inference
    * **TODO** read 
    * Transformers can do Bayesian inference, The propose prior-data fitted networks (PFNs). PFNs leverage large-scale machine learning techniques to approximate a larget set of posteriors (Muller 2021, muller2021transformers).
    * Requires the ability to sample from a prior distribution over supverised learning tasks (or functions). 
    * Their method restates the objective prosterior apprimixation as a supervised classification problem with set valued input: it repeatedly draws a task (or function) from the prior, draws a set of data points and their labels from it, marks on of the labels and learns to make probabilistic predictions for it based on the set-valued input of the rest of the data points.
    * PFNs can nearly perfectly mimic Gaussian Processes and also enable efficient Bayesian Inference for intractable problems, with 200-fold speedups in networks evaluated. 
    * PFNs perofrm well in GP regression, Bayesian NNs, classification on tabular data, few-shot iamge classification - there applications demonstrate generality of PFNs. 

nielsen2020survae
-----------------
    * SurVAE Flows: Surjections to Bridge the Gap between VAEs and Flows 
    * TODO [ ] read 

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

raissi2019physics
-----------------
    * Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations
    * Discussed by Bastiaan from 2022-09-14 - Deep Learning  

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

rasmussen2003gaussian
---------------------
    * Gaussian Processes in machine learning. 

restek2018high
--------------
    * Explanation of gas-chromatraphy in food science for FAMEs. 

riad2022learning
----------------
    * Learning strides in convolutional neural networks 

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

smart2005using
--------------
    * Using genetic programming for multiclass classification by simultaneously solving component binary classification problems 
    * Multi-class classification with Genetic Programs using a Classification Map (CM). 
    * Maps a float to a classification label using a classification map.
    * Create class boundaries sequentially on a floating point number line. 
    * If program output is within a class boundary, it belongs to that class. 
    * For multi-class classification, their is an identical interval of 1.0. 

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

tran2019genetic
---------------
    * Genetic programming for multiple-feature construction on high-dimensional Classification Data 
    * This paper includes an example of Multi-tree GP. 
    * I can apply Multi-tree GP for a one-vs-all multi-class classification problem. 

tegmark2020aifeynman
--------------------
    * AI Feynman: A physics-inspired method for symbolic regression
    * Tegmark et al. developed they AI Feynman \cite{udrescu2020ai}. 
    * This algorithm can derive physics equations from data using symbolic regression. 
    * Symbolic regression is a difficult task, but by simplifying properties exhibited by physics equations (i.e symmetry, composability, separability), the problem can be reduced. 
    * Their work uses blackbox neural networks, to derive interpretable models that can easily be verified by humans. 

tegmark2020aifeynman2.0
-----------------------
    * AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity
    * 2nd iteration for the AI Feynman 2.0. 
    * More robust towards noise and bad data. 
    * Can discover more formulas that previous method. 
    * Implements Normalizaing flows. 
    * Method for generalized symmetries (abitrary modularity in the compuational graph formula)

tegmark2021aipoincare
---------------------
    * AI Poincaré 2.0: Machine Learning Conservation Laws from Differential Equations
    * TODO [ ] READ 

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

vaswani2017attention
--------------------
     * Attention is all you need

Weinstein2022hunter 
-------------------
    * A Hunter Gatherer's Guide to the 21st Century (Book).
    * pg. 229 "Evolutionary stable strategy - A strategy incapable of invasion by competitors"

wolpert1997no
-------------
    * No free lunch theorum. 
    * No classification algorithm that beats the rest for every problem. 
    * As training instances approaches infinity, classification accuracy on all distributions of noise, approaches predicting mean class. 
    * All machine learning algorithms are task specific, don't generalize to all problems, no artifical general intelligence (AGI), yet... 

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

yang2022noise
-------------
    * Noise-Aware Sparse Gaussian Processes and Application to Reliable Industrial Machinery Health Monitoring
    * (Yang 2022) proposed a Noise-Aware Sparse Gaussain Process (NASGP) with Bayesian Inference Network. 
    * Data: 
        * Domain - maintainace of machinary equipment requires real-time health monitoring. Most state-of-the-art models require high quality monitoring data, but are not robust to noise present in real-world applications. 
        * Problem - predict an estimate of the reamining useful life of machinary equipment using noisy data. 
    * Method: 
        * Noise-Awate Sparse Gaussain Processes (NASGP) + Bayesian Inference Network. 
    * Results: 
        * NASGP are capable of high-performance and credible assessment under strong noises. 
        * Developed a generative additive model to bridge the gap between latent inference mechanism and domain expert knowledge. 
        * Method worked well in two different domains: (1) remaining useful life prognosis, (2) fault diagnosis in rolling bearings. 
    * Why it matters?
        * The method is robust to noise, and can be applied to real-world applications, not just academic benchmarks (toy datasets). 
        * Method provides a generative additive model that works well in two different domains.
        * Important to monitor machinary equipment in real-world applications, to ensure safety, automation, and efficiency.
    * See 2022-10-12 Deep Learning for more 

zemmal2016adaptative
--------------------
    * Adaptative S3VM Semi Supervised Learning with Features Cooperation for Breast Cancer Classification 
    * (Zemmal 2016) propose S3VM, a semi-supverised SVM, that combines labelled and unlablled datasets, to improve SVM performance for Breast Cancer Classification. 
    * Data: 
        * Domain - breast cancer is the most common cause of cancer and the second leading cause of all cancer deaths. Early detection in the intiial stages of cancer is crucial for survival.
        * Problem - Breast Cancer Classification with computer-aided diagnosis (CAD), to classify breast cancer tumours as malignant or benign. 
        * Collecting data is expensive and time-consuming, practitioners wait 5 years after treatment to label a patient as survived or not. Label annotation of the datasets is a slow processs, but their is an abundance of unlabelled data. 
    * Method: 
        * Supervised + unsupervised learning to boost SVM performance. 
        * Using unlabeleld data (unsupervised) to ensure the decision boundaries are drawn through low density areas. 
    * Results: 
        * Evaluate method by increasing the proportion of labelled-to-unlabelled data for each test (rarely, moderately low, moderate). 
        * Promising results were validated on a real-world dataset of 200 images. 
    * Why it matters? 
        * SVM performance can be improved by using unlabelled data.
        * Unlabelled data is abundant, but expensive to label. Methods that utilize unlabelled data are cheap, efficient, and improve performance. 
        * Early cancer detection is crucial for survival. 
    * TODO [ ] - read. 

zhang2008two
------------
    * Zhang et al. proposed a 2-D COW algorithm for aligning gas chromatography and mass spectrometry. 
    * The algorithm warps local regions of the data to maximise the correlation with known reference samples. 
    * This work uses data fusion with labelled reference samples, to improve the quality of new samples.

zhao2019maximum
---------------
    * Maximum relevance and minimum redundancy feature selection methods for a marketing machine learning platform. 
    * See (Ding 2005, ding2005minimum) for original MRMR paper. 
    * This (Zhao 2019) is a paper from Uber engineering. 
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
