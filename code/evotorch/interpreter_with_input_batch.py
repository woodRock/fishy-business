import logging
import torch 
from torch.nn import CrossEntropyLoss
from evotorch.tools.structures import CList
from typing import Iterable, Optional, Union
from interpreter import Interpreter

class InterpreterWithInputBatch:
    def __init__(
        self,
        *,
        max_stack_length: int,
        program_batch_size: int,
        input_size: int,
        input_batch_size: int,
        unary_ops: Iterable,
        binary_ops: Iterable,
        pass_means_terminate: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self._program_batch_size = int(program_batch_size)
        self._input_batch_size = int(input_batch_size)
        self._input_size = int(input_size)
        self._batch_size = self._program_batch_size * self._input_batch_size

        self._interpreter = Interpreter(
            max_stack_length=max_stack_length,
            batch_size=self._batch_size,
            input_size=self._input_size,
            unary_ops=unary_ops,
            binary_ops=binary_ops,
            pass_means_terminate=pass_means_terminate,
            device=device,
        )


    def run(self, program_batch: torch.Tensor, input_batch: torch.Tensor) -> torch.Tensor:
        programs = torch.repeat_interleave(program_batch, self._input_batch_size, dim=0)
        inputs = (
            input_batch
            .expand(self._program_batch_size, self._input_batch_size, self._input_size)
            .reshape(self._batch_size, self._input_size)
        )
        return (
            self._interpreter
            .run(programs, inputs)
            .reshape(self._program_batch_size, self._input_batch_size)
        )


    def compute_mean_squared_error(
        self,
        program_batch: torch.Tensor,
        input_batch: torch.Tensor,
        desired_output_batch: torch.Tensor
    ) -> torch.Tensor:
        output = self.run(program_batch, input_batch)
        return torch.mean((output - desired_output_batch) ** 2, dim=-1)
    
    
    def compute_categorical_cross_entropy(
        self,
        program_batch: torch.Tensor,
        input_batch: torch.Tensor,
        desired_output_batch: torch.Tensor
    ) -> torch.Tensor:
        """ 
        Categorical cross entropy loss for mulit-class classification task

        Returns the categorical cross entropy loss for a population of genetic programs.

        Args:
            program_batch (torch.tensor): a batch of genetic programs.
            input_batch (torch.tensor): a batch of input values.
            desired_output_batch (torch.tensor): a batch of desired output values.

        Returns:
            losses (torch.tensor): Categorical cross entropy loss for the population.
        """
        output = self.run(program_batch, input_batch)
        losses = []
        loss = CrossEntropyLoss(label_smoothing=0.1)
        for out in output:
            l = loss(out, desired_output_batch)
            losses.append(l)
        losses = torch.as_tensor(losses, dtype=torch.float32)
        return losses
    

    @property
    def stack(self) -> CList:
        return self._interpreter.stack


    @property
    def instructions(self) -> list:
        return self._interpreter.instructions


    @property
    def program_batch_size(self) -> int:
        return self._program_batch_size