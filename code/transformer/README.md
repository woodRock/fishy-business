# Transformer 

1. Devlin, J. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805. https://arxiv.org/abs/1810.04805
2. Vaswani, A. (2017). Attention is all you need. Advances in Neural Information Processing Systems. https://user.phil.hhu.de/~cwurm/wp-content/uploads/2020/01/7181-attention-is-all-you-need.pdf

# Experiments - without pre-trained weights

Species

```python
import numpy as np
train = np.array([
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
])
test = np.array([
    1.0000,
    1.0000,
    1.0000,
    0.9615,
    0.9615,
    1.0000,
    1.0000,
    1.0000,
    0.9615,
    1.0000,
])
mean, std = train.mean(), train.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
mean, std = test.mean(), test.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
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
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
])
test = np.array([
    0.8333,
    0.9000,
    0.9000,
    0.7500,
    0.8889,
    0.8333,
    0.9000,
    0.9000,
    0.7500,
    0.7500,
])
mean, std = train.mean(), train.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
mean, std = test.mean(), test.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
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
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
])
test = np.array([
    0.5884,
    0.5083,
    0.3725,
    0.5306,
    0.5111,
    0.5731,
    0.5595,
    0.4778,
    0.5067,
    0.4667,
])
mean, std = train.mean(), train.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
mean, std = test.mean(), test.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
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
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
])
test = np.array([
    0.8586,
    0.9327,
    1.0000,
    0.8246,
    0.9259,
    0.8333,
    0.9333,
    0.9744,
    0.8690,
    0.9394,
])
mean, std = train.mean(), train.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
mean, std = test.mean(), test.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
```

Instance-recognition hard:

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
])
test = np.array([
    0.2591,
    0.3889,
    0.3636,
    0.4194,
    0.3056,
    0.4273,
    0.3750,
    0.2667,
    0.3654,
    0.3214,
    0.3403,
    0.3690
])
mean, std = train.mean(), train.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
mean, std = test.mean(), test.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
```

# Experiments - with pre-trained weights

Species

```python
import numpy as np
train = np.array([
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
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
    0.9615,
    1.0000,
])
mean, std = train.mean(), train.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
mean, std = test.mean(), test.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
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
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
])
test = np.array([
    0.8333,
    0.9000,
    0.8333,
    0.8333,
    0.7778,
    0.9167,
    0.9000,
    0.9000,
    0.8333,
    0.6667,
])
mean, std = train.mean(), train.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
mean, std = test.mean(), test.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
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
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
])
test = np.array([
    0.6500,
    0.5850,
    0.5377,
    0.4611,
    0.4933,
    0.6300, 
    0.6057,
    0.4667,
    0.6275,
    0.5752,
])
mean, std = train.mean(), train.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
mean, std = test.mean(), test.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
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
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
])
test = np.array([
    0.8406,
    0.9333,
    0.9444,
    0.9373,
    0.9296,
    0.8414,
    0.9030,
    1.0000,
    0.9373,
    0.9296,
])
mean, std = train.mean(), train.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
mean, std = test.mean(), test.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
```

Instance-recognition hard:

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
])
test = np.array([
    0.3729,
    0.3269,
    0.3918,
    0.2828,
    0.2751,
    0.4053,
    0.3214,
    0.3403,
    0.3690,
])
mean, std = train.mean(), train.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
mean, std = test.mean(), test.std()
print(f"{mean*100:.2f}\% $\pm$ {std*100:.2f}\%")
```