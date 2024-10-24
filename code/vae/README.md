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
    0.9615,
    1.0000,
    1.0000,
    0.9615,
    1.0000,
    0.9615,
    1.0000,
    1.0000,
    1.0000, # 10
    1.0000,
    0.9615,
    1.0000,
    1.0000,
    0.9500,
    1.0000,
    0.9615,
    1.0000,
    1.0000,
    1.0000, # 20
    1.0000,
    1.0000,
    1.0000,
    0.9615,
    0.9500,
    1.0000,
    0.9615,
    1.0000,
    1.0000,
    0.9615, # 30
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
    0.9107 ])
test = np.array([
    0.8333,
    1.0000,
    0.8333,
    0.7500,
    0.6667,
    0.7500,
    0.9000,
    0.9000,
    0.6111,
    0.6667,
    0.6667,
    0.8333,
    0.8333,
    0.6111,
    0.5000, 
    0.7500,
    0.7500,
    0.8333,
    0.7500,
    0.4000, # 20
    0.8333,
    0.9000,
    0.7500,
    0.4444,
    0.7778,
    0.7500,
    0.8333,
    0.9000,
    0.8333,
    0.5833 ])
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
    0.5400,
    0.5162,
    0.5150,
    0.5992,
    0.3553,
    0.5105,
    0.5114,
    0.5625,
    0.5762,
    0.5265, # 10
    0.5197,
    0.4917,
    0.6838,
    0.5048,
    0.5405,
    0.5267,
    0.4721,
    0.5556,
    0.4257,
    0.5250, # 20
    0.6738,
    0.6617,
    0.5312,
    0.6377,
    0.4838,
    0.6000,
    0.6101,
    0.5278,
    0.6619,
    0.5625, # 30
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
    0.8586,
    0.9167,
    0.9327,
    0.9107,
    0.9394,
    0.8159,
    0.9116,
    0.9167,
    0.8487,
    0.9394, # 10
    0.8364,
    0.9019,
    0.9327,
    0.8813,
    0.9394,
    0.7741,
    0.9000,
    0.8974,
    0.8348,
    0.9394, # 20
    0.8364,
    0.9524,
    0.9667,
    0.9056,
    0.9394,
    0.8141,
    0.9410,
    0.9667,
    0.8611,
    0.9394, # 30
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
    0.5000,
    0.5204,
    0.5297,
    0.5010,
    0.5000,
    0.5000,
    0.5000,
    0.5000,
    0.5000,
    0.5000,
])
mean, std = train.mean(), train.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
mean, std = test.mean(), test.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
```