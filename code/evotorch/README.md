# Genetic Program in EvoTorch

This example is based off of the example on the EvoTorch documentation for Genetic Programming. That tutorial is available here: https://docs.evotorch.ai/v0.5.1/examples/notebooks/Genetic_Programming/?h=genetic

This code has been modified from a regression to a multi-class classification problem. With the loss function being swapped from Mean Squared Error to Categorical Cross Entropy.

## Evaluation

We can evaluate execution speed using the following command:

```bash
$ time python3 main.py -p 10 -g 5
```

Here are some code evaluation stats from using the `time` command as:

```md
# Cuda 11 GPU
real    0m25.191s
user    0m24.010s
sys     0m1.158s

# CPU woodj4
real    1m38.087s
user    16m37.399s
sys     1m53.307s
```