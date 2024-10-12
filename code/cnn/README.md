# Convolutional Neural Network 

This script lets you run a convolutional neural networks (CNN) code on the GPUs.

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

# Experiments

Species:

```python
import numpy as np
train = np.array([
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
])
test = np.array([
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
])
mean, std = train.mean(), train.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
mean, std = test.mean(), test.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
```

Part:

```python
import numpy as np
train = np.array([
    0.8833,
    0.8413,
    0.9097,
    1.0000,
    0.9722,
    0.9444,
    0.9190,
    0.9246,
    0.9524,
    0.9667,
])
test = np.array([
    0.6000,
    0.9000,
    1.0000,
    1.0000,
    1.0000,
    0.6250,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
])
mean, std = train.mean(), train.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
mean, std = test.mean(), test.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
```

Oil:

```python
import numpy as np
train = np.array([
    0.7609,
    0.9020,
    0.9320,
    0.8444,
    0.9000,
    0.8044,
    0.9556,
    0.9167,
    0.9345,
    0.9558,
])
test = np.array([
    0.6042,
    0.9238,
    1.0000,
    1.0000,
    1.0000,
    0.6580,
    0.9000,
    1.0000,
    1.0000,
    1.0000,
])
mean, std = train.mean(), train.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
mean, std = test.mean(), test.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
```

Cross-species:

```python
import numpy as np
train = np.array([
    0.9505,
    1.0000,
    0.9722,
    0.9841,
    0.9521,
    0.8903,
    0.9825,
    0.9683,
    0.9444,
    0.9671,
])
test = np.array([
    0.8246,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    0.8095,
    0.9667,
    1.0000,
    1.0000,
    1.0000,
])
mean, std = train.mean(), train.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
mean, std = test.mean(), test.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
```

Instance recogntion: 

```python
import numpy as np
train = np.array([
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
])
test = np.array([
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
])
train.mean(), train.std()
test.mean(), test.std()
```