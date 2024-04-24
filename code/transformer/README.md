# Transformer 

This script lets you run a transformer code on the GPUs.

## Running

Run the bask script as follows.

```bash
$ ./run.sh
```

## Task Spooler

Task Spooler was originally developed by Lluis Batlle i Rossell but is no longer maintained. The branch introduced here is a fork of the original program with more features including GPU support.

https://justanhduc.github.io/2021/02/03/Task-Spooler.html

```bash
$ ts -G <number-of-gpus> <command>
$ ts -G 1 python3 main.py
```