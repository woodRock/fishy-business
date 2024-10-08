# LSTM: Long Short-Term Memory

1. Hochreiter, S. (1997). Long Short-term Memory. Neural Computation MIT-Press. https://sophieeunajang.wordpress.com/wp-content/uploads/2020/10/lstm.pdf

# Experiments.

```python
import numpy as np
train = np.array([
    0.9850,
    1.0000,
    0.9960,
    0.9879,
    0.9990,
    0.9841,
    0.9803,
    0.9812,
    0.9732
])
test = np.array([
    0.6807050092764378,
    0.6923933209647495,
    0.6769944341372913,
    0.6633580705009277,
    0.6667903525046381,
    0.6601113172541744,
    0.7115027829313543,
    0.6546382189239333,
    0.7161410018552876
])
train.mean(), train.std()
test.mean(), test.std()
```