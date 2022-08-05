# Automated Fish Classification Using Unprocessed Fatty Acid Chromatographic Data: A Machine Learning Approach

This is the paper for the Australasian Joint Conference on Artificial Intelligence 2023 [AJCAI](https://ajcai2022.org/). 
The [PDF](https://github.com/woodRock/fishy-business/blob/main/paper/paper.pdf) is available. 

## Abstract 

Fish is approximately 40% edible fillet. 
The remaining 60% can be processed into low-value fertilizer or high-value pharmaceutical-grade omega-3 concentrates.
High-value manufacturing options depend on the composition of the biomass, which varies with fish species, fish tissue and seasonally throughout the year.
Fatty acid composition, measured by Gas Chromatography, is an important measure of marine biomass quality.
This technique is accurate and precise, but processing and interpreting the results is time-consuming and requires domain-specific expertise.
The paper investigates different classification and feature selection algorithms for their ability to automate the processing of Gas Chromatography data.
Firstly, the paper proposes a preprocessing imputation method for aligning timestamps in Gas Chromatography data.
Secondly, experiments found that SVM could classify compositionally diverse marine biomass based on raw chromatographic fatty acid data. 
The SVM model is interpretable through visualization which can highlight important features for classification.
Lastly, experiments demonstrated that applying feature selection significantly reduced dimensionality and improved classification performance on high-dimensional low sample-size datasets.
According to the reduction rate, feature selection could accelerate the classification system up to four times.
  