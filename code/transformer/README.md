# Transformer 

This script lets you run a transformer code on the GPUs.

## Running

To run experiments, you can use the `run.sh` script. This uses the task spooler `ts` command to run 10 experiments with parallel execution on a GPU server.

Example usage:

```bash
$ ./run.sh
```

To run an individual experiment, you can execute the following command. Say we want to perform genetirc programming on the species dataset `-d`, with 50 epochs `-e`.

Example usage:

```bash
$ python3 main.py -d species -e 50
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


# Experiments.

Instance recognition

```python
import numpy as np
train = np.array([
    0.5118,
    0.5181,
    0.7942,
    0.5481,
    0.5000,
    0.5180,
    0.5057,
    0.8573,
    0.5000,
    0.7233,
    0.5181,
])
test = np.array([
    0.7394,
    0.6334,
    0.8600,
    0.7269,
    0.6073,
    0.6693,
    0.6577,
    0.8598,
    0.5320,
    0.7233,
])
train.mean(), train.std()
test.mean(), test.std()
```