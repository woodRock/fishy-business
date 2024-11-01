# Genetic Programming in DEAP

## Run

To run experiments, you can use the `run.sh` script. This uses the task spooler `ts` command to run 10 experiments with parallel execution on a GPU server.

Example usage:

```bash
$ ./run.sh
```

To run an individual experiment, you can execute the following command. Say we want to perform genetirc programming on the species dataset `-d`, with 50 generations `-g` , and a population beta `-b` of 1.

Example usage:

```bash
$ python3 main.py -d species -g 50 -b 1
```

## Task Spooler

Task Spooler was originally developed by Lluis Batlle i Rossell but is no longer maintained. The branch introduced here is a fork of the original program with more features including GPU support.

Say we want to execute the main script on the GPU server, with one GPU `-G` per parallel exeuction of that script.

Example usage:

```bash
# ts -G <number-of-gpus> <command>
$ ts -G 1 python3 main.py
```

For more information on the `ts` command from task spooler see: https://justanhduc.github.io/2021/02/03/Task-Spooler.html

## Grid computing

The grid can be used to run many experiments at once using the `qsub` command. First, we need to ensure the grid computing environment is setup properly.

Run the following command:

```bash
$ need sgegrid
```

To run 30 experiments at once, we run the following command:

```bash
$ qsub -t 1-30:1 grid.sh 
```

For more information on the grid, see https://ecs.wgtn.ac.nz/Support/TechNoteEcsGrid#A_basic_job_submission_script

## DEAP 

For more information on DEAP, a python library for evolutionary compuation, see: https://deap.readthedocs.io/en/master/

# Experiments

Species:

```python
import numpy as np
train = np.array([
    0.989583,
    0.989583,
    0.986842,
    1.0000,
    0.989583,
])
test = np.array([
    1.0000,
    0.575000,
    0.875000,
    0.833333,
    0.763889,
])
train.mean(), train.std()
test.mean(), test.std()
```

Part:

```python
import numpy as np
train = np.array([
    0.925000,
    0.866667,
    0.900000,
    0.916667,
    0.802778,
])
test = np.array([
    0.200000,
    0.333333,
    0.833333,
    0.000000,
    0.000000,
])
train.mean(), train.std()
test.mean(), test.std()
```

Oil:

```python
import numpy as np
train = np.array([
    0.537415,
    0.554422,
    0.616327,
    0.466667,
    0.528571,
])
test = np.array([
    0.142857,
    0.166667,
    0.154762,
    0.345238,
    0.107143,
])
train.mean(), train.std()
test.mean(), test.std()
```

Cross-species:

```python
import numpy as np
train = np.array([
    0.863182,
    0.752437,
    0.864888,
    0.860755,
    0.804843,
])
test = np.array([
    0.675926,
    0.674074,
    0.822222,
    0.407407,
    0.712963,
])
train.mean(), train.std()
test.mean(), test.std()
```