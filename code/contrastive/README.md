# Contrastive

## Running in the background

To run the optuna scripts in the background on an ssh server use the following command.

```bash
nohup python3 contrastive/run_optuna.py transformer --n_trials 200 --timeout 36000 > output.log 2>&1 &
```

## Results

| Encoder | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Test Loss | Test Accuracy |
|---|---|---|---|---|---|---|
| cnn | 3.2431 | 0.7629 | 3.5370 | 0.9750 | 3.3108 | 0.7771 |
| ensemble | 3.2427 | 0.7742 | 3.3338 | 0.9583 | 3.3681 | 0.6792 |
| kan | 3.4076 | 0.5571 | 3.4340 | 0.5000 | 3.4340 | 0.5000 |
| mamba | 3.4019 | 0.5542 | 3.4019 | 0.6042 | 3.3916 | 0.5063 |
| moe | 3.2660 | 0.7758 | 3.3504 | 0.9458 | 3.3927 | 0.7042 |
| rcnn | 3.3994 | 0.5588 | 3.4423 | 0.7188 | 3.4151 | 0.5000 |
| transformer | 3.2689 | 0.7713 | 3.3654 | 0.9563 | 3.3810 | 0.7083 |
| vae | N/A | 0.5113 | N/A | 0.5000 | N/A | 0.5000 |
