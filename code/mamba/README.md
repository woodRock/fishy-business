# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

1. Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752. https://arxiv.org/abs/2312.00752

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
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    0.9615,
    0.9545,
    1.0000,
    0.9500, # 10
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
    0.7500,
    0.9000,
    0.7500,
    0.8889,
    0.6000,
    0.9000,
    0.8333,
    0.8333,
    0.7778,
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
    0.6429,
    0.5738,
    0.5833,
    0.5850,
    0.5833,
    0.6131,
    0.6119,
    0.4857,
    0.5952,
    0.3548,
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
    0.8444,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    0.8274,
    1.0000,
    1.0000,
    1.0000,
    1.0000
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
    0.9942,
    0.9879,
    0.9942,
    0.9881,
    1.0000,
    0.9881,
    0.9879,
    1.0000,
    1.0000,
    0.9704,
])
test = np.array([
    0.5715213358070501,
    0.5796846011131725,
    0.6241187384044526,
    0.6003710575139146,
    0.6064935064935065,
    0.6169758812615955,
    0.6593692022263451,
    0.5620593692022263,
    0.5569573283858997,
    0.6034322820037106,
])
train.mean(), train.std()
test.mean(), test.std()
```