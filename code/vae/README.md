# VAE: Variational Autoencoder

1. Kingma, D. P. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114. https://www.ee.bgu.ac.il/~rrtammy/DNN/StudentPresentations/2018/AUTOEN~2.PDF

# Experiments

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