# Contrastive

## Running in the background

To run the optuna scripts in the background on an ssh server use the following command.

```bash
nohup python3 contrastive/run_optuna.py transformer --n_trials 200 --timeout 36000 > output.log 2>&1 &
```