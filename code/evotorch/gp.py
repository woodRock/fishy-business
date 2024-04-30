import logging
import torch
from operators import AdditionalTorchFunctions
from evotorch.operators import TwoPointCrossOver
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from evotorch import Problem
from functools import partial
from data import load_dataset
from problem import ProgramSynthesisProblem

def mutate_programs(problem: Problem, mutation_rate: float, programs: torch.Tensor) -> torch.Tensor:
    num_instructions = len(problem.instructions)
    mutate = torch.rand(programs.shape, device=programs.device) < mutation_rate
    num_mutations = int(torch.count_nonzero(mutate))
    result = programs.clone()
    mutated = torch.randint(0, num_instructions, (num_mutations,), device=programs.device)
    result[mutate] = mutated
    return result


def target_function(inputs: torch.Tensor) -> torch.Tensor:
    x = inputs[:, 0]
    y = inputs[:, 1]
    return AdditionalTorchFunctions.binary_div(x + y, torch.cos(x)) + torch.sin(y)


class GeneticProgram():

    def __init__(self, 
                population=234, 
                generations=50, 
                dataset="species",
                crossover_rate=0.8,
                mutation_rate=0.2,
                ) -> None:
        """ Genetic Program implemented in EvoTorch.
        
        Args: 
            population (int): the population size.
            generations (int): the number of generations to train for.
            dataset (str): Fish "species" or "part". Defaults "species".
            crossover_rate (float): the probability of crossover. Defaults to 0.8
            mutation_rate (float): the probability of mutation. Defaults to 0.2
        """
        self.population = population
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.dataset = dataset
        self.X, self.y = load_dataset(dataset)

    def __call__(self):
        logger = logging.getLogger(__name__)

        inputs = self.X
        outputs = self.y

        inputs = torch.as_tensor(inputs, dtype=torch.float32)
        outputs = torch.as_tensor(outputs, dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        program_length = 2
        if self.dataset == "species":
            program_length = 2 
        elif self.dataset == "part":
            program_length = 6

        self.problem = ProgramSynthesisProblem(
            inputs=inputs,
            outputs=outputs,
            unary_ops=[torch.neg, torch.sin, torch.cos, AdditionalTorchFunctions.unary_div],
            binary_ops=[torch.add, torch.sub, torch.mul, AdditionalTorchFunctions.binary_div],
            program_length=program_length,
            device=device,
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
        )

        StdOutLogger(ga)
        ga.run(self.generations)

        best_solution = ga.status["best"]
        logger.info(f"best solution: {best_solution}")

        logger.info(f"self.problem:  {self.problem}")

        logger.info(f"problem.instruction_dict: {self.problem.instruction_dict}")