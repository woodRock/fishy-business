# Siamese network 

## Genetic Programming

Run the following command to evaluate performance of Genetic Programming for contrastive learning.

```bash
python3 gp.py -nt 30 -g 200 -p 100
```

# Experiments

## CNN

Instance recognition: contrastive learning 

```python
import numpy as np
train = np.array([
    0.9538,
    0.9371,
    0.9585,
    0.9693,
    0.9445,
    0.9508,
    0.9471,
    0.9557,
    0.9444,
    0.9456,
])
test = np.array([
    0.7449,
    0.6250,
    0.7137,
    0.7075,
    0.6348,
    0.6106,
    0.6743,
    0.6202,
    0.6628,
    0.6867,
])
train.mean(), train.std()
test.mean(), test.std()
```

## RCNN

Instance recognition: contrastive learning 

```python
import numpy as np
train = np.array([
    1.0000,
    1.0000,
    0.9996,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    0.9991,
    1.0000,
])
test = np.array([
    0.7965,
    0.7060,
    0.7165,
    0.7826,
    0.8062,
    0.7090,
    0.7107,
    0.7131,
    0.7303,
    0.8318,
])
train.mean(), train.std()
test.mean(), test.std()
```

## KAN 

Instance recognition: contrastive learning 

```python
import numpy as np
train = np.array([
    0.9322,
    0.9164,
    0.9472, 
    0.9553,
    0.9643,
    0.9528,
    0.9453,
    0.9562,
    0.9593,
    0.9184,
])
test = np.array([
    0.7098,
    0.7710,
    0.7098,
    0.7526,
    0.7529,
    0.8288,
    0.7469,
    0.7648,
    0.7178,
    0.7540,
    0.7715,
])
train.mean(), train.std()
test.mean(), test.std()
```

## LSTM

Instance recognition: contrastive learning 

```python
import numpy as np
train = np.array([
    0.9848,
    0.9917,
    0.9883,
    0.9914,
    0.9834,
    0.9878,
    0.9923,
    0.9780,
    0.9774,
    0.9900,
])
test = np.array([
    0.7529,
    0.7685,
    0.6841,
    0.6474,
    0.7623,
    0.7975,
    0.8294,
    0.8104,
    0.7656,
    0.6869,
])
train.mean(), train.std()
test.mean(), test.std()
```

## Mamba

Instance recognition: contrastive learning 

```python
import numpy as np
train = np.array([
    0.9289,
    0.9450,
    0.9424,
    0.9438,
    0.9726,
    0.9579,
    0.9591,
    0.9491,
    0.9554,
    0.9698
])
test = np.array([
    0.7435,
    0.6800,
    0.7709,
    0.7852,
    0.7614,
    0.7682,
    0.7819,
    0.7633,
    0.7866,
    0.8557
])
train.mean(), train.std()
test.mean(), test.std()
```

## Transformer

Instance recognition: contrastive learning 

```python
import numpy as np
train = np.array([
    0.9857,
    0.9697,
    0.9581,
    0.9816,
    0.9793,
    0.9779,
    0.9846,
    0.9740,
    0.9818,
    0.9852,
])
test = np.array([
    0.7898,
    0.7317,
    0.7557,
    0.7440,
    0.7091,
    0.6887,
    0.8270,
    0.7174,
    0.8042,
    0.8022,
])
train.mean(), train.std()
test.mean(), test.std()
```

## VAE

Instance recognition: contrastive learning 

```python
import numpy as np
train = np.array([
    0.9480,
    0.9716,
    0.9723,
    0.9689,
    0.9621,
    0.9618,
    0.9698,
    0.7994,
    0.9620,
])
test = np.array([
    0.6493,
    0.7054,
    0.6328,
    0.6985,
    0.6431,
    0.7270,
    0.6929,
    0.6660,
    0.6905,
])
train.mean(), train.std()
test.mean(), test.std()
```

##  GP

Instance recognition: contrastive learning 

```python
 
                    Validation: 
import numpy as np
train = np.array([
    0.8771,
    0.9140,
    0.8091,
    0.8870,
    0.8726,
    0.8473,
    0.8584,
    0.9054,
    0.7199,
    0.8450,
])
test = np.array([
    0.6702,
    0.5841,
    0.5460,
    0.5946,
    0.6910,
    0.5123,
    0.6805,
    0.6959,
    0.5503,
    0.6022,
])
train.mean(), train.std()
test.mean(), test.std()
```

## GP error

```
Traceback (most recent call last):
  File "/home/woodj/Desktop/fishy-business/code/siamese/gp.py", line 338, in <module>
    main()
  File "/home/woodj/Desktop/fishy-business/code/siamese/gp.py", line 335, in main
    model.train(train_data, val_data)
  File "/home/woodj/Desktop/fishy-business/code/siamese/gp.py", line 303, in train
    record = stats.compile(pop)
  File "/home/woodj/.local/lib/python3.10/site-packages/deap/tools/support.py", line 201, in compile
    values = tuple(self.key(elem) for elem in data)
  File "/home/woodj/.local/lib/python3.10/site-packages/deap/tools/support.py", line 201, in <genexpr>
    values = tuple(self.key(elem) for elem in data)
  File "/home/woodj/Desktop/fishy-business/code/siamese/gp.py", line 274, in <lambda>
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
IndexError: tuple index out of range
```