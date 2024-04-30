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
        X_train: Iterable,
        y_train: Iterable,
        X_val: Iterable,
        y_val: Iterable,
        X_test: Iterable,
        y_test: Iterable,
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
                
        self._X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
        self._y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device)
        self._X_val = torch.as_tensor(X_val, dtype=torch.float32, device=device)
        self._y_val = torch.as_tensor(y_val, dtype=torch.float32, device=device)
        self._X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        self._y_test = torch.as_tensor(y_test, dtype=torch.float32, device=device)

        self._X_train_batch_size, self._X_train_size = self._X_train.shape
        [y_train_batch_size] = self._y_train.shape
        assert y_train_batch_size == self._X_train_batch_size

        self._X_val_batch_size, self._X_val_size = self._X_val.shape
        [y_val_batch_size] = self._y_val.shape
        assert y_val_batch_size == self._X_val_batch_size

        self._X_test_batch_size, self._X_test_size = self._X_test.shape
        [y_test_batch_size] = self._y_test.shape
        assert y_test_batch_size == self._X_test_batch_size

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

    def _set_interpreter(self, num_programs, input_size, input_batch_size) -> None:
        self._interpreter = InterpreterWithInputBatch(
            max_stack_length=self._program_length,
            program_batch_size=num_programs,
            input_size=input_size,
            input_batch_size=input_batch_size,
            unary_ops=self._unary_ops,
            binary_ops=self._binary_ops,
            pass_means_terminate=self._pass_means_terminate,
            device=self._X_train.device,
         )

    def _get_interpreter(self, num_programs: int) -> InterpreterWithInputBatch:
        if (self._interpreter is None) or (num_programs > self._interpreter.program_batch_size):
             self._interpreter = InterpreterWithInputBatch(
                max_stack_length=self._program_length,
                program_batch_size=num_programs,
                input_size=self._X_train_size,
                input_batch_size=self._X_train_batch_size,
                unary_ops=self._unary_ops,
                binary_ops=self._binary_ops,
                pass_means_terminate=self._pass_means_terminate,
                device=self._X_train.device,
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

        # Run the problem over the training, validation, and test datasets.
        batch.set_evals(interpreter.compute_balanced_accuracy(programs, self._X_train, self._y_train)[:num_programs])
        self._set_interpreter(self._X_val_size, self._X_val_size, self._X_val_batch_size)
        batch.set_evals(interpreter.compute_balanced_accuracy(programs, self._X_val, self._y_val)[:num_programs])
        self._set_interpreter(self._X_test_size, self._X_test_size, self._X_test_batch_size)
        batch.set_evals(interpreter.compute_balanced_accuracy(programs, self._X_test, self._y_test)[:num_programs])

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
