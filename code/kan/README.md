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
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    0.8333,
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
    0.7556,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    0.6667,
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
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    0.8648,
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

Instance recognition:

```python
import numpy as np
train = np.array([
    1.0, 
    0.9881, 
    0.9881, 
    0.9815, 
    0.9826, 
    0.9881, 
    0.9826, 
    0.9934, 
    0.9942, 
    0.9823
])
test = np.array([
    0.6333951762523191, 
    0.609369202226345, 
    0.6721706864564008, 
    0.5659554730983303,  
    0.5115027829313543, 
    0.550834879406308, 
    0.5860853432282004, 
    0.5189239332096475, 
    0.6459183673469387, 
    0.6346011131725418,
])
train.mean(), train.std()
test.mean(), test.std()
```