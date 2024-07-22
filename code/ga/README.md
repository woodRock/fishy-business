# Genetic Algorithm (GA)

## Run

To run experiments, you can use the `run.sh` script. This uses the task spooler `ts` command to run 10 experiments with parallel execution on a GPU server.

Example usage:

```bash
$ ./run.sh
```

To run an individual experiment, you can execute the following command. Say we want to perform genetirc programming on the species dataset `-d`, with 50 generations `-g` , and a population beta `-b` of 1.

Example usage:

```bash
$ python3 main.py -d species -g 50 -b 1
```

## Task Spooler

Task Spooler was originally developed by Lluis Batlle i Rossell but is no longer maintained. The branch introduced here is a fork of the original program with more features including GPU support.

Say we want to execute the main script on the GPU server, with one GPU `-G` per parallel exeuction of that script.

Example usage:

```bash
# ts -G <number-of-gpus> <command>
$ ts -G 1 python3 main.py
```

For more information on the `ts` command from task spooler see: https://justanhduc.github.io/2021/02/03/Task-Spooler.html

## Grid computing

The grid can be used to run many experiments at once using the `qsub` command. First, we need to ensure the grid computing environment is setup properly.

Run the following command:

```bash
$ need sgegrid
```

To run 30 experiments at once, we run the following command:

```bash
$ qsub -t 1-30:1 grid.sh 
```

For more information on the grid, see https://ecs.wgtn.ac.nz/Support/TechNoteEcsGrid#A_basic_job_submission_script



## DEAP 

For more information on DEAP, a python library for evolutionary compuation, see: https://deap.readthedocs.io/en/master/