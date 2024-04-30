import logging
import torch
from evotorch import Problem, SolutionBatch
from typing import Iterable, Optional, Union
from interpreter_with_input_batch import InterpreterWithInputBatch

class ProgramSynthesisProblem(Problem):
    def __init__(
        self,
        unary_ops: Iterable,
        binary_ops: Iterable,
        inputs: Iterable,
        outputs: Iterable,
        program_length: int,
        pass_means_terminate: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        num_actors: Optional[Union[str, int]] = None,
        num_gpus_per_actor: Optional[Union[str,int]] = None,    
    ):
        if device is None:
            device = torch.device("cpu")
        else:
            device = torch.device(device)

        self._program_length = int(program_length)
        self._inputs = torch.as_tensor(inputs, dtype=torch.float32, device=device)
        self._outputs = torch.as_tensor(outputs, dtype=torch.float32, device=device)

        self._input_batch_size, self._input_size = self._inputs.shape
        [output_batch_size] = self._outputs.shape
        assert output_batch_size == self._input_batch_size

        self._unary_ops = list(unary_ops)
        self._binary_ops = list(binary_ops)
        self._pass_means_terminate = pass_means_terminate

        self._interpreter: Optional[InterpreterWithInputBatch] = None
        num_instructions = len(self._get_interpreter(1).instructions)

        super().__init__(
            objective_sense="max",
            solution_length=self._program_length,
            dtype=torch.int64,
            bounds=(0, num_instructions - 1),
            device=device,
            num_actors=num_actors,
            num_gpus_per_actor=num_gpus_per_actor,
            store_solution_stats=True,
        )

    def _get_interpreter(self, num_programs: int) -> InterpreterWithInputBatch:
        if (self._interpreter is None) or (num_programs > self._interpreter.program_batch_size):
            self._interpreter = InterpreterWithInputBatch(
                max_stack_length=self._program_length,
                program_batch_size=num_programs,
                input_size=self._input_size,
                input_batch_size=self._input_batch_size,
                unary_ops=self._unary_ops,
                binary_ops=self._binary_ops,
                pass_means_terminate=self._pass_means_terminate,
                device=self._inputs.device,
            )
        return self._interpreter

    def _evaluate_batch(self, batch: SolutionBatch):
        num_programs = len(batch)
        interpreter = self._get_interpreter(num_programs)

        if num_programs < interpreter.program_batch_size:
            programs = torch.zeros(
                (interpreter.program_batch_size, self.solution_length),
                dtype=torch.int64,
                device=interpreter.stack.device
            )
            programs[:num_programs, :] = batch.values
        else:
            programs = batch.values

        batch.set_evals(interpreter.compute_balanced_accuracy(programs, self._inputs, self._outputs)[:num_programs])

    @property
    def instructions(self) -> list:
        interpreter = self._get_interpreter(1)
        return interpreter.instructions

    @property
    def instruction_dict(self) -> dict:
        result = {}
        for i_instruction, instruction in enumerate(self.instructions):
            result[i_instruction] = instruction
        return result
