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
    0.9615,
    0.9545,
    0.9500,
    1.0000,
    1.0000,
    0.9615,
    0.9545,
    0.9615,
    1.0000, # 10
    1.0000,
    0.9615,
    0.9167,
    0.9615,
    1.0000,
    1.0000,
    1.0000,
    0.9545,
    0.8606,
    1.0000, # 20
    1.0000,
    0.9615,
    0.9545,
    0.9091,
    0.9615,
    1.0000,
    0.9615,
    0.9545,
    0.9615,
    1.0000, # 30
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
    0.7500,
    0.7500,
    0.6250,
    0.6111,
    0.5000,
    0.6250,
    0.8333,
    0.4444,
    0.8750,
    0.5000, # 10
    0.6667,
    0.7500,
    0.5000,
    0.8750,
    0.5000,
    0.6250,
    0.9000,
    0.8333,
    0.8333,
    0.6250, # 20
    0.7000,
    0.9000,
    0.7500,
    0.8750,
    0.7000,
    0.7000,
    0.7500,
    0.7500,
    0.8750,
    0.5000, # 30
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
    0.6095,
    0.5992,
    0.4673,
    0.5625,
    0.4676,
    0.6081,
    0.5476,
    0.5833,
    0.6187,
    0.4503, # 10
    0.6630,
    0.5151,
    0.5625,
    0.5909,
    0.4524,
    0.5988,
    0.6321,
    0.7121,
    0.5622,
    0.5102, # 20
    0.5903,
    0.5500,
    0.5278,
    0.5870,
    0.5323,
    0.6300,
    0.5754,
    0.4910,
    0.4556,
    0.4711, # 30
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
    0.8171,
    0.8545,
    0.9074,
    0.8301,
    0.8810,
    0.8030,
    0.8578,
    0.8069,
    0.7083,
    0.7875, # 10
    0.8167,
    0.8377,
    0.8167,
    0.7611,
    0.9167,
    0.8333,
    0.8205,
    0.9091,
    0.7667,
    0.8095, # 20
    0.8095,
    0.8821,
    0.8814,
    0.7885,
    0.8974,
    0.7296,
    0.8545,
    0.8860,
    0.7399,
    0.8974, # 30
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