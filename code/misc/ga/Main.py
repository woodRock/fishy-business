"""
Main - Main.py
==============

This is the main routine for a Genetric Algorithm (GA) model.

Using GA we solve (1) onemax, and (2) 2nd-order polynomial equations.
GA implements evaluation, selection and reproduction on a bitstring representations of the solution.
Evaluation measures the performance of the solution by the objective function.
We perform tournament selection to decide which individuals are kepts.
The reproduction is done by crossover and mutation.

References:
    1.  Simple Genetic Algorithm From Scratch in Python, Jason Brownlee,
        Machine Learning Mastery https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/

TODO:
DEAP Github - https://github.com/DEAP/deap
"""

# TODO
# [x] Complete the tutorial.
# [ ] Apply this to GC fish classification task.
# [ ] Try basic DEAP algorithm.
# [ ] Implement elitism.

from numpy.random import randint, rand
from .plot import plot_error
import numpy as np


def selection(pop, scores, k=3):
    """
    Tournament selection.

    Args:
        pop: Population.
        scores: Scores.
        k: Number of participants.

    Returns:
        Selection individual.
    """
    # First random selection.
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # Chec if better (e.g. perform a tournament).
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def mutation(bitstring, mutation_rate):
    """
    Mutation.

    Args:
        individual: Individual bitstring to be mutated.
        mutation_rate: The probability a mutation will occur.
    """
    for i, _ in enumerate(bitstring):
        # Check for a mutation.
        if rand() < mutation_rate:
            # Flip the bit.
            bitstring[i] = 1 - bitstring[i]


def crossover(p1, p2, crossover_rate):
    """
    Crossover.

    Args:
        p1: Parent 1.
        p2: Parent 2.
        crossover_rate: The probability a crossover between two parents occurs.

    Returns:
        Children.
    """
    # Children are copies of parents by default.
    c1, c2 = p1.copy(), p2.copy()
    # Check for recombination.
    if rand() < crossover_rate:
        # Select crossover poin that is not on the end of the string.
        pt = randint(1, len(p1)-2)
        # Perform crossover.
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


def onemax(x):
    """
    A onexmax objective function (boring).

    With the OneMax problem you are searching for a bitstring where all values are filled with ones.

    Args:
        x: An individual.

    Returns:
        The objective function value.
    """
    return -sum(x)


def objective(x):
    """
    2nd order polynomial objective funciton (interesting).

    Args:
        x: An individual.

    Returns:
        The objective function value.
    """
    return x[0]**2.0 + x[1]**2.0


def decode(bounds, n_bits, bitstring):
    """
    Decodes a bitstring into a real number.

    Args:
        bounds: Bounds.
        n_bits: Number of bits.
        bitstring: Bitstring.

    Returns:
        Decoded number.
    """
    decoded = []
    largest = 2**n_bits
    for i, _ in enumerate(bounds):
        # Extract the substring.
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = bitstring[start:end]
        # Convert bitstring to a string of chars.
        chars = ''.join([str(s) for s in substring])
        # Convert string to integer.
        integer = int(chars, 2)
        # Scale integer to desired range.
        value = bounds[i][0] + (integer/largest) * \
            (bounds[i][1] - bounds[i][0])
        # Store
        decoded.append(value)
    return decoded


def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, crossover_rate, mutation_rate):
    """
    Genetic algorithm.

    Args:
        objective: Objective function.
        n_bits: Number of bits.
        n_iter: Number of iterations.
        n_pop: Number of population.
        crossover_rate: The probability a crossover occurs.
        mutation_rate: The probability a mutation occurs.

    Returns:
        Best solution.
    """
    errors = []
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best, best_eval = 0, objective(pop[0])

    # enumerate generations
    for gen in range(n_iter):
        # Decode population.
        decoded = [decode(bounds, n_bits, p) for p in pop]
        # evaluate all candidates in the population
        scores = [objective(d) for d in decoded]

        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(f">{gen}, new best f({decoded[i]}) = {scores[i]}")
                errors.append(best_eval)
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = []

        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, crossover_rate):
                # mutation
                mutation(c, mutation_rate)
                # store for next generation
                children.append(c)
        # replace population
        pop = children

    plot_error(errors)
    return [best, best_eval]


if __name__ == "__main__":
    bounds = [[-5.0, 5.0], [-5.0, 5.0]]
    iterations = 1000
    n_bits = 16
    n_pop = 100
    crossover_rate = 0.9
    mutation_rate = 1.0 / (float(n_bits) * len(bounds))
    best, score = genetic_algorithm(
        objective, bounds, n_bits, iterations, n_pop, crossover_rate, mutation_rate)
    print("Done!")
    decoded_best = decode(bounds, n_bits, best)
    print(f"Best solution: {decoded_best}.\nScore: {score}.")
