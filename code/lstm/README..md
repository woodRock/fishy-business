# LSTM: Long Short-Term Memory

1. Hochreiter, S. (1997). Long Short-term Memory. Neural Computation MIT-Press. https://sophieeunajang.wordpress.com/wp-content/uploads/2020/10/lstm.pdf

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
])
test = np.array([
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
])
train.mean(), train.std()
test.mean(), test.std()
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
])
test = np.array([
    0.7500,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
])
train.mean(), train.std()
test.mean(), test.std()
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
])
test = np.array([
    0.6607,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
])
train.mean(), train.std()
test.mean(), test.std()
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
])
test = np.array([
    0.8364,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
])
train.mean(), train.std()
test.mean(), test.std()
```