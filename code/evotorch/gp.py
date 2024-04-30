import logging
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split
from torcheval.metrics import MulticlassAccuracy
from operators import AdditionalTorchFunctions
from evotorch.operators import TwoPointCrossOver
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger, PandasLogger
from evotorch import Problem
from functools import partial
from data import load_dataset
from problem import ProgramSynthesisProblem

def mutate_programs(problem: Problem, mutation_rate: float, programs: torch.Tensor) -> torch.Tensor:
    """
    Perform the mutation operator on the population of programs.

    Args:
        problem (Problem): the problem contains the instruction list.
        mutation_rate (float): the probability for mutation to occur.
        programs (torch.Tensor): a list of programs in the population.

    Returns
        result (torch.Tensor): the mutated population is given.
    """
    num_instructions = len(problem.instructions)
    mutate = torch.rand(programs.shape, device=programs.device) < mutation_rate
    num_mutations = int(torch.count_nonzero(mutate))
    result = programs.clone()
    mutated = torch.randint(0, num_instructions, (num_mutations,), device=programs.device)
    result[mutate] = mutated
    return result


class GeneticProgram():

    def __init__(self, 
                population=1023, 
                generations=100, 
                dataset="species",
                crossover_rate=0.8,
                mutation_rate=0.2,
                elitism=True,
                num_actors=1,
                num_gpus_per_actor=1,
                file_path = "figures/evolutionary_process.png",
                ) -> None:
        """ Genetic Program implemented in EvoTorch.
        
        Args: 
            population (int): the population size.
            generations (int): the number of generations to train for.
            dataset (str): Fish "species" or "part". Defaults "species".
            crossover_rate (float): the probability of crossover. Defaults to 0.8
            mutation_rate (float): the probability of mutation. Defaults to 0.2
            elitism (bool): Keep best individuals from each generation. defaults to True.
            num_actors (int): Number of GPUs to run on. Defaults to 3 GPUs.
            num_gpus_per_actor (float): the number of GPUs per actor.
        """
        self.population = population
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.num_actors = num_actors
        self.num_gpus_per_actor = num_gpus_per_actor
        self.dataset = dataset
        self.X, self.y = load_dataset(dataset)
        self.file_path = file_path

    def __call__(self):
        """
        Run the Genetic Program.

        This code executes the genetic program to solve the problem.
        The dataset can be specified as fish "species" or "part".
        The program returns the best invidiual, and the instruction set for reference.
        """
        logger = logging.getLogger(__name__)

        # Optionally fix the generator for reproducible results
        generator = torch.Generator().manual_seed(42)
        X_train, X_val, X_test = random_split(self.X, [0.4, 0.3, 0.3], generator=generator)
        y_train, y_val, y_test = random_split(self.y, [0.4, 0.3, 0.3], generator=generator)
        
        X_train = torch.as_tensor(X_train, dtype=torch.float32)
        y_train = torch.as_tensor(y_train, dtype=torch.float32)

        device = torch.device("cuda" if (torch.cuda.is_available() and self.num_actors == 1) else "cpu")
        # Ray communicates between actors on the CPU, it is recommended that when you have num_actors > 1
        # source: https://docs.evotorch.ai/v0.5.1/user_guide/problems/
        # device = "cpu"
    
        program_length = 2
        if self.dataset == "species":
            program_length = 2 
        elif self.dataset == "part":
            program_length = 6
        else:
            raise ValueError(f"Incorrect dataset specification: {self.dataset}")

        self.problem = ProgramSynthesisProblem(
            inputs=X_train,
            outputs=y_train,
            unary_ops=[torch.neg, torch.sin, torch.cos, AdditionalTorchFunctions.unary_div],
            binary_ops=[torch.add, torch.sub, torch.mul, AdditionalTorchFunctions.binary_div],
            program_length=program_length,
            device=device,
            num_actors=self.num_actors,
            num_gpus_per_actor=self.num_gpus_per_actor,
        )

        ga = GeneticAlgorithm(
            self.problem,
            operators=[
                    TwoPointCrossOver(
                        self.problem, 
                        tournament_size=4,
                        cross_over_rate=self.crossover_rate), 
                    partial(mutate_programs, self.problem, self.mutation_rate)],
            re_evaluate=False,
            popsize=self.population,
            elitist=self.elitism
            )

        StdOutLogger(ga)
        pandas_logger = PandasLogger(ga)
        ga.run(self.generations)

        progress = pandas_logger.to_dataframe()
        progress.mean_eval.plot()
        plt.savefig(self.file_path)
        # Evaluate on the validation set.
        self.problem = ProgramSynthesisProblem(
            inputs=X_val,
            outputs=y_val,
            unary_ops=[torch.neg, torch.sin, torch.cos, AdditionalTorchFunctions.unary_div],
            binary_ops=[torch.add, torch.sub, torch.mul, AdditionalTorchFunctions.binary_div],
            program_length=program_length,
            device=device,
            num_actors=self.num_actors,
            num_gpus_per_actor=self.num_gpus_per_actor,
        )
        ga.run(1)

        # Evaluate on the test set.
        self.problem = ProgramSynthesisProblem(
            inputs=X_test,
            outputs=y_test,
            unary_ops=[torch.neg, torch.sin, torch.cos, AdditionalTorchFunctions.unary_div],
            binary_ops=[torch.add, torch.sub, torch.mul, AdditionalTorchFunctions.binary_div],
            program_length=program_length,
            device=device,
            num_actors=self.num_actors,
            num_gpus_per_actor=self.num_gpus_per_actor,
        ) 
        ga.run(1)

        # Take the best solution and record it to a logging file.
        best_solution = ga.status["best"]
        logger.info("Below is the best solution encountered so far")
        logger.info(f"best solution: {best_solution}")
        logger.info("This Genetic Program is for the following problem:")
        logger.info(f"self.problem:  {self.problem}")
        logger.info("The program reported above can be analyzed with the help of this instruction set:")
        logger.info(f"problem.instruction_dict: {self.problem.instruction_dict}")
