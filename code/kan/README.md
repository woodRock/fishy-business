# KAN: Kolmogorov-Arnold Networks

1. Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., ... & Tegmark, M. (2024). Kan: Kolmogorov-arnold networks. arXiv preprint arXiv:2404.19756. https://arxiv.org/abs/2404.19756 

# Experiments.

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
    1.0000,
    0.9615,
    1.0000,
    1.0000,
    0.9615,
    1.0000,
    0.9615,
    0.9286, # 10
    1.0000,
    0.9615,
    1.0000,
    0.9615,
    0.9286,
    1.0000,
    0.9615,
    1.0000,
    0.9615,
    0.9615, # 20
    1.0000,
    0.9615,
    1.0000,
    0.9615,
    0.9286,
    1.0000,
    0.9615,
    1.0000,
    0.9500,
    0.9500, # 30
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
    0.8333,
    0.8333,
    0.7500,
    0.7500,
    0.7222,
    0.6000,
    0.7500,
    0.6000,
    0.6250,
    0.8333, # 10
    0.6667,
    0.7500,
    0.6000,
    0.6000,
    0.8889,
    0.7500,
    0.7000,
    0.8333,
    0.7000,
    0.8889, # 20
    0.6667,
    0.7500,
    0.7500,
    0.6111,
    0.8889,
    0.6250,
    0.7000,
    0.7500,
    0.6111,
    0.8889, # 30
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
    0.6944,
    0.6310,
    0.4693,
    0.5867,
    0.6024,
    0.5889,
    0.6906,
    0.5546,
    0.7114,
    0.4000, # 10
    0.6286,
    0.5317,
    0.5288,
    0.7230,
    0.5833,
    0.7101,
    0.6500,
    0.5826,
    0.6533,
    0.5833, # 20 
    0.6190,
    0.5722,
    0.4274,
    0.8033,
    0.3726,
    0.7627,
    0.6167,
    0.5833,
    0.6111,
    0.3278, # 30
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
    0.8487,
    0.9190,
    0.9697,
    0.8611,
    0.9296,
    0.7889,
    0.8915,
    1.0000,
    0.8238,
    0.9394, # 10
    0.8082,
    0.9524,
    1.0000,
    0.8834,
    0.9296,
    0.8082,
    0.9153,
    1.0000,
    0.8586,
    0.9394, # 20
    0.8082,
    0.9524,
    1.0000,
    0.9077,
    0.9667,
    0.7929,
    0.9333,
    1.0000,
    0.9077,
    0.9394, # 30
])
mean, std = train.mean(), train.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
mean, std = test.mean(), test.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
```

Instance recognition:

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
    1.0000
])
test = np.array([
    0.6262,
    0.5838,
    0.6231,
    0.5871,
    0.5871,
    0.6706,
    0.6335,
    0.6065,
    0.6109,
    0.6119,
])
mean, std = train.mean(), train.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
mean, std = test.mean(), test.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
```