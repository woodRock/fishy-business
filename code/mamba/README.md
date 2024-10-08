# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

1. Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752. https://arxiv.org/abs/2312.00752

# Experiments

Instance recognition

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