# Genetic Program

## Requirements 

These are the external libraries required to run this code. 

```bash 
pip install numpy matplotlib deap
```

## Execution 

The code can be executed with the following command

```bash 
$ python3 -m gp.Main
``` 

## Results 

### Single-Tree GP 

This is an example of a Single Tree GP for classification. 

![Single Tree GP](./assets/SingleTree_GP.png)

### Multi-Tree GP 

These are example trees for a one-vs-all multi-class classifier. 

Here is the tree for bluecod. 

![Multi Tree GP - Bluecod](./assets/MultiTree_GP_BCO.png)

Here is the tree for gurnard. 

![Multi Tree GP - Gluecod](./assets/MultiTree_GP_GUR.png)

Here is the tree for snapper. 

![Multi Tree GP - Snapper](./assets/MultiTree_GP_SNA.png)

Here is the tree for tarakihi. 

![Multi Tree GP - Tarakihi](./assets/MultiTree_GP_TAR.png)