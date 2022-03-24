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