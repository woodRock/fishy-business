# Transformer 

1. Devlin, J. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805. https://arxiv.org/abs/1810.04805
2. Vaswani, A. (2017). Attention is all you need. Advances in Neural Information Processing Systems. https://user.phil.hhu.de/~cwurm/wp-content/uploads/2020/01/7181-attention-is-all-you-need.pdf

# Experiments

Species

```python
import numpy as np
train = np.array([
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
])
test = np.array([
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
])
train.mean(), train.std()
test.mean(), test.std()
```

Part

```python
import numpy as np
train = np.array([
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
])
test = np.array([
    0.8333,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
])
train.mean(), train.std()
test.mean(), test.std()
```

Oil

```python
import numpy as np
train = np.array([
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
])
test = np.array([
    0.6297,
    0.9714,
    1.0000,
    1.0000,
    0.8538,
    0.8538,
])
train.mean(), train.std()
test.mean(), test.std()
```

Cross-species

```python
import numpy as np
train = np.array([
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
])
test = np.array([
    0.8333,
    0.9744,
    1.0000,
    1.0000,
    1.0000,
])
train.mean(), train.std()
test.mean(), test.std()
```