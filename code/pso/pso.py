"""
PSO - PSO.y 

This module contains a Particle Swarm Optimization (PSO) for feature selection of Gas Chromatography data. 
The PSO mimicks the social behaviour of a flock of birds, or school of fish. 
Referred to as a swarm, because unlike birds, we particles allow collisions, where to particles occupy the same position. 
This is a Wrapper-based PSO, that measures fitness as a balance of SVM classification accuracy and selection ratio. 

References:
1. Kennedy, J., & Eberhart, R. (1995, November). Particle swarm optimization. 
    In Proceedings of ICNN'95-international conference on neural networks 
    (Vol. 4, pp. 1942-1948). IEEE.

"""

import numpy as np
import matplotlib.pyplot as plt
from .problem import FeatureSelection


class Particle:

    def __init__(self, length, max_pos, min_pos, max_vel, min_vel, w, c1, c2, problem):
        self.length = length
        self.max_pos = max_pos
        self.min_pos = min_pos
        self.max_vel = max_vel
        self.min_vel = min_vel
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.problem = problem
        self.position = min_pos + np.random.rand(length) * (max_pos - min_pos)
        self.velocity = np.zeros(length)
        self.fitness = self.problem.worst_fitness()
        self.pbest_pos = np.zeros(length)
        self.pbest_fit = self.problem.worst_fitness()
        self.gbest_pos = np.zeros(length)
        self.gbest_fit = self.problem.worst_fitness()

    def update(self):
        self.velocity = self.w * self.velocity + \
            self.c1 * np.random.rand(self.length) * (self.pbest_pos - self.position) + \
            self.c2 * np.random.rand(self.length) * \
            (self.gbest_pos - self.position)

        self.velocity[self.velocity < self.min_vel] = self.min_vel
        self.velocity[self.velocity > self.max_vel] = self.max_vel
        self.position = self.position + self.velocity
        self.position[self.position < self.min_pos] = self.min_pos
        self.position[self.position > self.max_pos] = self.max_pos


class Swarm:

    def __init__(self, n_particle, length, problem, n_iterations,
                 max_pos, min_pos, max_vel, min_vel, verbose=False):
        self.verbose = verbose
        self.n_particle = n_particle
        self.prob = problem
        self.n_iterations = n_iterations

        self.pop = [Particle(length,
                             max_pos=max_pos, min_pos=min_pos,
                             max_vel=max_vel, min_vel=min_vel,
                             w=0.72984, c1=1.496172, c2=1.496172, problem=self.prob)
                    for _ in range(n_particle)]

        # Used for visualization.
        self.history = []

    def iterate(self):
        for i in range(self.n_iterations):
            new_w = 0.9 - i * (0.9 - 0.4) / self.n_iterations
            gbest_fit = self.pop[0].gbest_fit
            gbest_index = self.pop[0].gbest_pos
            gbest_updated = False
            for index, par in enumerate(self.pop):
                par.w = new_w
                par.fitness = self.prob.fitness(par.position)
                if self.prob.is_better(par.fitness, par.pbest_fit):
                    par.pbest_fit = par.fitness
                    par.pbest_pos = np.copy(par.position)
                if self.prob.is_better(par.pbest_fit, gbest_fit):
                    gbest_fit = par.pbest_fit
                    gbest_index = index
                    gbest_updated = True
            if gbest_updated:
                for par in self.pop:
                    par.gbest_fit = self.pop[gbest_index].pbest_fit
                    par.gbest_pos = np.copy(self.pop[gbest_index].pbest_pos)
            for par in self.pop:
                par.update()

            # Append the best fitness for each iteration.
            self.history.append(gbest_fit)

        return self.pop[0].gbest_pos, self.pop[0].gbest_fit

    def plot_fitness(self, title=None):
        """Plots the fitness of the PSO over time.
        """
        plt.figure()
        plt.plot(self.history)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title('Evolutionary Process' if title is None else title)
        plt.show()


def pso(X, y):
    """ Convinience wrapper for PSO, which returns selected features for best individual. 

    Args:   
        X: the feature set. 
        y: the class labels. 

    Returns:
        sel_fea: The selected features ndexes. 
    """
    prob = FeatureSelection(minimized=True, X=X, y=y)
    pop_size = 30
    n_iterations = 100
    no_fea = X.shape[1]
    swarm = Swarm(n_particle=pop_size, length=no_fea, n_iterations=n_iterations,
                  max_pos=1.0, min_pos=0.0, max_vel=0.2, min_vel=-0.2,
                  problem=prob)
    best_sol, best_fit = swarm.iterate()
    sel_fea = np.where(best_sol > prob.threshold)[0]
    return sel_fea
