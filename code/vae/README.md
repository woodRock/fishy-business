# VAE: Variational Autoencoder

1. Kingma, D. P. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114. https://www.ee.bgu.ac.il/~rrtammy/DNN/StudentPresentations/2018/AUTOEN~2.PDF

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
    0.8024,
    0.8532,
    0.7238,
    0.8524,
    0.9048,
    0.8000,
    0.9375,
    0.8413,
    0.9167,
    0.9107,
])
test = np.array([
    0.7500,
    1.0000,
    1.0000,
    0.9000, 
    0.7500,
    0.7500,
    1.0000,
    1.0000,
    0.9000,
    0.8333,
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
    0.5810,
    0.6341,
    0.6081,
    0.6876,
    0.7536,
    0.3438,
    0.6161,
    0.4748,
    0.7069,
    0.6841,
])
test = np.array([
    0.5546,
    0.6865,
    0.7933,
    0.6957,
    0.7017,
    0.5784,
    0.6289,
    0.8556,
    0.7381,
    0.8056,
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
    0.7444,
    0.9676,
    0.9872,
    0.9861,
    1.0000,
    0.8610,
    0.9210,
    0.9861,
    0.8778,
    1.0000
])
test = np.array([
    0.9406,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    0.8021,
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

Instance recognition

```python
import numpy as np
train = np.array([
    0.5003,
    0.5005,
    0.5097,
    0.5000,
    0.5000,
    0.5000,
    0.5000,
    0.5000,
    0.5000,
    0.5000,
])
test = np.array([
    0.546660482374768,
    0.5581632653061225,
    0.5265306122448979,
    0.5,
    0.5530612244897959,
    0.5173469387755102,
    0.5153061224489796,
    0.536734693877551,
    0.5244897959183673,
    0.5112244897959184
])
train.mean(), train.std()
test.mean(), test.std()
```