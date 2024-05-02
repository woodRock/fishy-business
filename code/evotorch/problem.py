import torch
from evotorch import Problem, SolutionBatch
from typing import Iterable, Optional, Union
from interpreter_with_input_batch import InterpreterWithInputBatch

class FishClassificationProblem(Problem):
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
        minimized: bool = False,
    ):
        """
        Fish classification problem.

        Args: 
            unary_ops (list): the unary operators.
            binary_ops (list): the binary operators.
            X_train (torch.tensor): the input features.
            y_train (torch.tensor): the class labels.
            X_val (torch.tensor): the input features for the validation set.
            y_val (torch.tensor): the class labels for the validation set.
            X_test (torch.tensor): the input features for the test set.
            y_test (torch.tensor): the class labels for the test set.
            program_length (int): The length of the program's output.
            pass_means_terminate (bool): pass means terminate. Defaults to True.
            device (Optional(str, torch.device)): the device to run cuda on.
            num_actors (Optional(str,int)): the number of actors for Ray.
            num_gpus_per_actor (Optional(str,int)) the number of GPUs per actor.
            minimized (bool): the fitness function is either minimized or maximized.

        Example usage:

        ```python
        problem = FishClassificationProblem(
            inputs=[1,2,3,4,5],
            outputs=[2,4,6,8,10],
            unary_ops=[torch.neg, torch.sin, torch.cos, AdditionalTorchFunctions.unary_div],
            binary_ops=[torch.add, torch.sub, torch.mul, AdditionalTorchFunctions.binary_div],
            program_length=1,
            device="cpu",
            num_actors=3,
            num_gpus_per_actor=0.5,
        )
        ```
        """
        if device is None:
            device = torch.device("cpu")
        else:
            device = torch.device(device)

        self._program_length = int(program_length)        
        self._X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
        self._y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device)
        self._input_batch_size, self._input_size = self._X_train.shape
        [output_batch_size] = self._y_train.shape
        assert output_batch_size == self._input_batch_size

        self._X_val = torch.as_tensor(X_val, dtype=torch.float32, device=device)
        self._y_val = torch.as_tensor(y_val, dtype=torch.float32, device=device)
        self._input_batch_size, self._input_size = self._X_val.shape
        [output_batch_size] = self._y_val.shape
        assert output_batch_size == self._input_batch_size

        self._X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        self._y_test = torch.as_tensor(y_test, dtype=torch.float32, device=device)
        self._input_batch_size, self._input_size = self._X_test.shape
        [output_batch_size] = self._y_test.shape
        assert output_batch_size == self._input_batch_size
    
        self._unary_ops = list(unary_ops)
        self._binary_ops = list(binary_ops)
        self._pass_means_terminate = pass_means_terminate

        self._interpreter: Optional[InterpreterWithInputBatch] = None
        num_instructions = len(self._get_interpreter(1).instructions)

        self._minimized = minimized

        super().__init__(
            objective_sense="min" if self._minimized else "max",
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
                device=self._X_train.device,
            )
        return self._interpreter

    def _evaluate_batch(self, batch: SolutionBatch, verbose=False):
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

        batch_evals = interpreter.compute_balanced_accuracy(programs, self._X_train, self._y_train)[:num_programs]
        batch.set_evals(batch_evals)

        # Display the balanced accuracy for each dataset.
        # Only for verbose evaluation on the training, validation and test datasets.
        # It is expensive to perform this operation, so only use it once, at the end of training.
        if verbose:
            batch_evals = interpreter.compute_balanced_accuracy(programs, self._X_train, self._y_train)[:num_programs]
            # print(f"batch_evals: {batch_evals}")
            median, mean, best, worst = torch.median(batch_evals), torch.mean(batch_evals), torch.max(batch_evals), torch.min(batch_evals)
            print(f"\tTraining\n\t\tmedian: {median} \n\t\tmean: {mean:.4f}\n\t\tbest: {best:.4f}\n\t\tworst:{worst:.4f}")

            batch_evals = interpreter.compute_balanced_accuracy(programs, self._X_val, self._y_val)[:num_programs]
            median, mean, best, worst = torch.median(batch_evals), torch.mean(batch_evals), torch.max(batch_evals), torch.min(batch_evals)
            print(f"\tValidation\n\t\tmedian: {median} \n\t\tmean: {mean:.4f}\n\t\tbest: {best:.4f}\n\t\tworst:{worst:.4f}")
            
            batch_evals = interpreter.compute_balanced_accuracy(programs, self._X_test, self._y_test)[:num_programs]
            median, mean, best, worst = torch.median(batch_evals), torch.mean(batch_evals), torch.max(batch_evals), torch.min(batch_evals)
            print(f"\tTest\n\t\tmedian: {median} \n\t\tmean: {mean:.4f}\n\t\tbest: {best:.4f}\n\t\tworst:{worst:.4f}")


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
