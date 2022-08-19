Thoughts
========

2022-03-17 - Brains I 
---------------------

raine1997brain
    * Different brain chemistry can result in unwanted behaviour (murder) (Rain 1997).
    * The structure of the model limits is ability to understand abstract concepts. 
    * Humans with abnormal pre-frontal cortex lack the ability to make goal-oriented decisions. 
    * Reinforcment learning, job scheduling and other AI problems also fail at capturing the complexity of delayed gratification.
    * Perhaps, the circuitry involved in a functioning pre-frontal cortex can be translated into reinforcment learning. 

craik1972levels 
    * There are different levels of depth of processing (Craik 1972) .
    * Strengthening connections to existing neurons make it easier to encode information in long-term memory. 
    * We can build knowledge easier from existing context, no need to reinvent the wheel. 
    * This translates to LSTM models, pre-training (i.e. BERT), Thousand Brain Theory, etc. 
    * Models with deeper processing are more likely to be able synthesis information for challenging unseen data. 
    * Deeper processing involves elaborative rehearsal, which makes use of existing synapses, rather than training from scratch each time. 

craik1975depth
    * Deep processing leads to better recall (Craik 1975). 
    * A brain is able to remember more if we associate semantic meaning and context with the task. 
    * For a task, semantic meaning and context can remain static, while new training instances are dynamic. 
    * In machine learning, we can combine natural language processing (NLP) with existing tasks for greater model representation.
    * Examples include the combination of metadata NLP and computer vision for image dataset problems like classification, segmentation, detection. 

chase1973perception
    * Model representations from domain experts fail on illogical scenarios (Chase 1973). 
    * This work shows that the human brain cannot generalize on noise.
    * The analogy provides a concrete example of intelligence in humans that also translates to artificaical intelligence. 
    * This is a limitation of existing machine learning models, they struggle to perform well on unseen data. 

2022-03-18 - EC 
---------------
    * Sexual selection in Genetic Programming. Female preference (up, horizontal), Male (not fussy). Already done (Miller 2005)
    * Evolution in a dynamic environment + evolution can't reverse (hills and mountains analogy)
        * i.e. for multi-class, run many generations with two classes, then later add more. 
        * Slowly introduce features to the training process. 
    * Semantic distance use to eliminate redundant candidate solutions, use to assesss diversity of population and prune (see Tegmark 2020).
    * Imposing apriori belief structures on evolutionary computation - memes or culutral transmission (Dawkins 1976) to complement evolution.
        * Generalizing this, impose dogma on a model, based on domain expertise. 
        * Genetic operators become dogma, monogamy, polygamy, polyamory - only certain indiviudals carry these cultural ideas. 
        * Limit genetic operators to each individual - not all available to entire population at once. 
        * The hand the draws itself (Hofstadter 1979).

2022-03-21 - Cars 
-----------------
    * Follow up work on (Cai 2020, Codevilla 2018) needed: 
    * A rally car with two models, (1) driver and (2) directions giver.
    * Directions giver: 
        * A model that usges computer vision to create a map of the road from multiple cameras. 
        * Provides real-time plain speech instructions on how to traverse upcoming corners. 
        * "Sharp turn left in 500m"
    * Driver: 
        * Takes real-time plain speech language and performs driving tasks. 
        * Aims for maximum velocity and drift. 
    * Due to game theoretic directions and driver model, can generalize to unseen data (i.e. drive on new roads). 
    * Two independent tasks that can be optimized seperately - similar to a GAN with its critique and generator. 

2022-02-22 - Brains II 
----------------------

mantyla1998cue
    * People have better recall when they have multiple retrieval cues personal to them. 
    * Similar to how model-free reinforcement learning (Cai 2020, Akkaya 2019) exceeds human designed controllers. 
    * Recall is improved when connections between existing neurons are strengthened. 
    * Effective recall requires everyone to learn a model that is unique and personal to them. 

eich1975state, miles1998state, godden1975context
    * State and context dependent recall.
    * More likely to remember something given the same state and context. 
    * Similar to noise distributions, a model trained with gaussian noise will perform well on test set of same distribution. 
    * It would perform poorly with a different distribution of noise or none at all. 
    * The context and state are noise to the sensory registers that attempt to encode semantic meaning into short-term memory. 
    * Our brain can't seperate all noise from signal, so the context/state is partially encoded into our mental representation. 
    * Without that noise present, is becomes difficult to achieve the same activation pattern in the associated network. 
    * "Our virtues and our failings are inseparable, like force and matter. When they separate, man is no more" - Nikola Tesla

eyesenck1980effects
    * Value of distinct examples 
    * Similar to Deren Browns mind palace - linked to the Ancient greek technique for memorization. 
    * If he wants to remember something, he associates it with a graphic or lewd imagery - i.e. An queen of hearts, is a naked woman with her heart ripped out. 

brewer2006brown
    * We remember certain events with a distinct clarity and vivideness. 
    * We remember lightbulb events longer than other events, but still not well. 
    * We have a false confidence that these distinct events are accurate remembered. 
    * Each time we retell the story, we re-write it, and mis-remember it, the memory is foggyier than we think.
    * The metric of strong positive/negative emotion, while correlated, doesn't guarantee the accuracy of the memory. 
    * Similar to how a model may also share a false confidence of its accuracy on an distinct outlier.

2022-03-23 - Project Management
-------------------------------
    * Project management is important for self-supervised learning such as this PhD. 
    * I have created an agile kanban board to keep a public record of my work https://github.com/woodRock/fishy-business/projects/1 
    * We can use Github issues to link code commits to issues, and create user stories that achieve milestones. 
    * Basically, apply agile methodology from PMBOK to my studies, to avoid the Parkinson's law 
        * Parkinson's law - Parkinson's law is the adage that "work expands so as to fill the time available for its completion." (https://en.wikipedia.org/wiki/Parkinson%27s_law)
    * My proposal and thesis are the first two major milestones that I can consider at this early stage. 
    * As I get stuck into implementation and actual work, many more milestones will arise. 

2022-03-24 - Bayesian Optimization 
----------------------------------
    * shahriari2015taking and brochu2010tutorial were suggested readings from the FASLIP talk. 
    * Bayesian optimization through Gaussian Processes (GP) is an effective method for approximating prohibitive objective functions. 
    * Those functions may be prohibitive, because they are intractable, or computationally expensive, or other reasons. 
    * We used GP in the Conditional Neural Processes (Garnelo 2018, garnelo2018conditional) when building a the cnpRIR for Summer Research Project. 
    * The term neuro-evolutionary, used in (Eiben 2015 - eiben2015evolutionary), are hybrid algorithms that combine neural networks and evolutionary computation.

2022-03-31 - The Big Reset 2.0 
------------------------------
    * Based on FASLIP video https://fishy-business.readthedocs.io/en/latest/minutes.html#id10
    * When naming these things, they just pour fuel on the fire of conspiracy theorists. 
    * China uses technology/AI as a hammer, and when your tool is a hammer, everything starts to look like a nail. 
    * Some things don't need deep learning or AI for solutions - reduce depedency on technology can be a good things. 
    * WeChat is used for everything, messenger, social media, payments - when your phone dies you are no longer a citizen. 
    * Similar to digital vaccination pass system in New Zealand, when you phone dies you forfeit your rights as a human being. 
    * AI prosthetics are cool, but still not available en-masse, the rich and  continue their already shelted and privelaged lifestyle. We could give these to everyone, but there is no economic incentive. 
    * Even kids notice the Narrow-AI / free lunch theorum in Artificial Intelligence.
    * Theres an AI solution for every unique problem (e.g. Siri, Alexa, Spot, GPT-3, AlphaFold), but no ONE AI that can generalize for everthing (AGI).
    * A noticeable divide between aritificial intelligence in an academic sense and industry applications for these technologies. 
    * As adoption of AI increases ( **cough** the academics enter these industries) the diffusion of innovation can continue. 

2022-03-31 - CNN Results 
------------------------
    * Implemented the 1-dimensional CNN from ENGR489 - with some tweaks and adjustments. 
        * Assess the accuracy using k-fold stratified cross-validation (k = 10). 
        * This allows for direct comparison to the other classifiers (i.e. SVM, KNN, DT, ...). 
        * Manually hyperparameters tuning achieve 98% accuracy on fish species training set. 
        * Competitive with the Linear SVM classifier results (approx. 98.33%). 
    * Manual tuning for the CNN hyperparameters is a time consuming and laborious process. 
        * This was black magic and I was very lucky to find hyper-parameters that work well. 
        * I would like to explore using EC to perform a neural architecture search. 
        * Automate the process of tuning 
        * May provide improved performance over the SVM classifier for more difficult fish part dataset. 
        * The SVM provides an excellent baseline for the competitive performance of the desired CNN model.
        * SVM is far less computationally expensive to evaluate.
    * Paper / research 
        * For now, simple neural architecture search, which encodes batch size, epochs, and filter size for convolutional layers, and dropout probability. 
        * Later include a genetic algorithm that constructs the network design - but this remains beyond my initial scope. 
        * If results are promising, a paper on "Evolutionary Computation for Neural Architecture Search in Fish Oil Analysis" would be appropriate.

2022-04-04 - Chemical Stuff 
---------------------------
    * MegaSYN - computation proof of concept for using AI to manufacture biological weapons https://bit.ly/3x2enuY
        * Adjusted objective function for MegaSYN and used deadly training data. 
        * Fine-tuned a model to generate a synthetic compound similar to a deadly nerge agent. 
        * Reversed parameter, from minimization to maximiation for toxicity factor in objective function. 
        * Model was able to synthesize deadly biological toxins. 
        * AI has no sense of morality, it is up to the practitioner to remain ethical in their approach.  
    * Evolutionary Model of Varient Effect (EVE) - https://bit.ly/3K6GTPP
        * Model identifies mutations associated with disease in unlabelled datasets.  
        * Unsupervised clustering of chemical data, but then employing VAEs to assess liklihoods. 
        * Trained a VAE for each proteing family, given one variant in a protein family, it learned to compute liklihood of each amino acid in the sequence. 
        * Employing VAEs on unsupervised clustering algorithms combines clustering and auto-encoders. 
        * Model was more accurate that lab test results (i.e. 99% > 95% AUC)

2022-04-07 - CNN Fish Part 
--------------------------
    * CNN classification results for fish part. 
    * Fish part needs a simpler neural network to achieve high accuracy on fish part. 
    * Network needs a 90% dropout rate, but can achieve 84% accuracy. 
    * Still need to manually tune this model more to be competitive with the SVM.
    * LeakyRelu activation function improves performance on both datasets.

2022-04-22 - e2e with SCARA 
---------------------------
    * Could encorporate model-free reiforcement learning for simulation/real-world SCARA robot to take measurements in a factory. 
    * A general purpose AI robot controller to automate the data collection of the GC-MS / REIM data. 
    * This is a way to explore my interest in robotics / AI / computer vision - and encorporate this into my work. 
    * Overall goal: propose an end-to-end engineering solution to quaility assurance in food science. 
    * Moon-shot: (1) Data Collection, (2) Data Pre-processing, (3) Classification/Regression/Clustering, (4) Visualisation / NLP knowledge generated, (5) Verfication by Domain Expertise.
    * Try apply Aritifical intelligence techniques to automate all the processes above. 

2022-04-27 - Teleoperation SCARA 
---------------------------------
    * The paper (quin 2022, qin2022one) shows possibility for teleoperation of SCARA via a laptop webcam. 
    * This requires no special, 3D sensor setup, and can be done with a simple webcam - available to anyone with a laptop. 
    * This could be implemented to allow for remote operation of our SCARA for employees working from home (WFH). 
    * Furthermore, this could be similar to Tesla FSD, where human intervention can resolve edge cases. 
    * This dramatically increases the viability of the SCARA product for collecting REIMS data. 

2022-04-28 - AI Day + SCRUM Board 
---------------------------------
    * Who says you can't enjoy your PhD research https://twitter.com/jrhwood/status/1519514331200643072 
    * I presented the AI Day supercut to the FASLIP research group, and it was recieved very positively. 
    * This was a moment of Jungian Synchronicity - a self actualising and deeply meaningful experience. 
    * Side note: I made a gigantic prop replica of the SCRUM board from Silicon Valley https://www.youtube.com/watch?v=oyVksFviJVE 
    * This is achieves a major PhD research goal that I set for my ENGR391 Work Experience paper. 
    * I have successfully applied Project Management techniques to my study. 
    * I added my own twist, by using playing cards as a ranking/priority heuristic for ordering tasks.
    
2022-05-11 - First GP 
---------------------
    * I wrote my first Genetic Program (GP) from scratch today. 
    * We implement a tutorial by Jason from Machine Learning Mastery https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
    * It solved (1) onemax, and (2) 2nd-order polynomial equations. 
    * GP implements evaluation, selection and reproduction on a bitstring representations of the solution.
    * Evaluation measures the performance of the solution by the objective function.
    * We perform tournament selection to decide which individuals are kepts. 
    * The reproduction is done by crossover and mutation.

2022-05-27 - COVID gap 
----------------------
    * I had COVID-19, was very sick, and self-isolated.
    * This explains the gap, as you will, in my work here.

2022-06-17 - Importance of Writing 
----------------------------------
    * Writing is an important part of research, we refine our ideas, and correct our incorrect assumptions, we organize our thoughts. 
    * For example, when re-writing my draft paper, I realized my understanding of treated fish biomass waste was incorrect.
    * Something I would have never realized if I did not explicitly have to re-write my introduction section.

2022-08-18 - Github Copilot 
--------------------------- 
    * GitHub CoPiliot is an AI powered autocompletion code tool made using GPT-3 https://github.com/features/copilot/ 
    * I was lucky enough to test the software during its closed beta period. 
    * GitHub Copilot is available for free to developers with the Student License. 
    * As a PhD student, I qualify for this license for the next three years. 
    * I simply sent my Student ID and university email, and they verified I am a current student. 

2022-08-19 - Biology in EC 
--------------------------
    * Read this https://www.nature.com/articles/s42256-020-00278-8 
    * Genotype: the genetic makeup of an individual, i.e. the recessive gene for ginger hair. 
    * Phenotype: the expression of a trait (or gene), i.e. Ginger hair. 
    * These concepts from biology apply to Evolutionary Computation: 
        - Genetype is the encoding for an individual, the representation, i.e. A GP tree or binary string. 
        - Phenotype is that genotype after being decoded, the prediction, i.e. the output of the GP tree.