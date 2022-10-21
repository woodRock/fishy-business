# Code

This folder contains documentated codebases used to produce the results of the experiments presented in this thesis.

## Organisation

This folder is organized into seperate modules for each codebase:

```
.
├── cnn
├── data
├── ga
├── gp
├── pca
└── pso
```

With the exception of the data folder, which is used by all modules, each module is designed to function in isolation. Those modules are:

- [**cnn**](https://github.com/woodRock/fishy-business/tree/main/code/cnn), an 1D convolutional neural network (CNN) for classification on Gas Chromatography data.
- [**data**](https://github.com/woodRock/fishy-business/tree/main/code/data), this folder contains the Gas Chromatography data, stored as a matlab file. And the unprocessed raw data for archival purposes.
- [**ga**](https://github.com/woodRock/fishy-business/tree/main/code/ga), an implementation of a basic Genetic Algorithm (GA), that can solve basic 2nd order polynomials and onesum.
- [**gp**](https://github.com/woodRock/fishy-business/tree/main/code/gp), a Genetic Program (GP) for classification on the Fish Oil Data. It constructs a basic GP tree with arithmetic operators and a classification map (CM) to facilitate classification.  
- [**julia**](https://github.com/woodRock/fishy-business/tree/main/code/julia), a Julia implementation of PSO for the michalewicz & mccormick combinatorial optimization problems.
- [**pca**](https://github.com/woodRock/fishy-business/tree/main/code/pca), Principal Component Analysis (PCA) implemented from scratch using Eigenvectors and Numpy library. 
- [**pso**](https://github.com/woodRock/fishy-business/tree/main/code/pso), the feature selection codebase for the experiments for the AJCAI [paper](https://github.com/woodRock/fishy-business/tree/main/paper). This includes an implementation of Particle Swarm Optimisation.
