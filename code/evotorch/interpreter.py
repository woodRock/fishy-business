import torch
from typing import Iterable, Optional, Union
from evotorch.tools.structures import CList
from instruction import Instruction

class Interpreter:
    def __init__(
        self,
        *,
        max_stack_length: int,
        batch_size: int,
        input_size: int,
        unary_ops: Iterable,
        binary_ops: Iterable,
        pass_means_terminate: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            device = torch.device("cpu")
        else:
            device = torch.device(device)

        self._batch_size = int(batch_size)
        self._input_size = int(input_size)
        self._max_stack_length = int(max_stack_length)

        self._stack = CList(
            max_length=self._max_stack_length,
            batch_size=self._batch_size,
            dtype=torch.float32,
            device=device,
            verify=False,
        )
        self._inputs = torch.zeros(self._batch_size, self._input_size, dtype=torch.float32, device=device)

        self._instructions = []
        self._pass_means_terminate = bool(pass_means_terminate)

        for operation in ("pass", "swap", "duplicate"):
            self._instructions.append(
                Instruction(
                    inputs=self._inputs,
                    stack=self._stack,
                    arity=0,
                    operation=operation,
                )
            )

        for i_input in range(self._input_size):
            self._instructions.append(
                Instruction(
                    inputs=self._inputs,
                    stack=self._stack,
                    arity=0,
                    input_slot=i_input,
                )
            )

        for unary_op in unary_ops:
            self._instructions.append(
                Instruction(
                    inputs=self._inputs,
                    stack=self._stack,
                    arity=1,
                    function=unary_op,
                )
            )

        for binary_op in binary_ops:
            self._instructions.append(
                Instruction(
                    inputs=self._inputs,
                    stack=self._stack,
                    arity=2,
                    function=binary_op,
                )
            )

    @property
    def instructions(self) -> list:
        return self._instructions

    @property
    def stack(self) -> CList:
        return self._stack

    def run(self, program_batch: torch.Tensor, input_batch: torch.Tensor) -> torch.Tensor:
        self._stack.clear()
        program_batch = torch.as_tensor(program_batch, dtype=torch.int64, device=self._stack.device)
        batch_size, program_length = program_batch.shape
        assert batch_size == self._batch_size

        if self._pass_means_terminate:
            program_running = torch.ones(batch_size, dtype=torch.bool, device=self._stack.device)
        else:
            program_running = None

        self._inputs[:] = input_batch

        for t in range(program_length):
            instruction_codes = program_batch[:, t]

            if self._pass_means_terminate:
                program_running = program_running & (instruction_codes != 0)

            for i_instruction in range(1, len(self._instructions)):
                instruction_codes_match = (instruction_codes == i_instruction)
                if self._pass_means_terminate:
                    instruction_codes_match = instruction_codes_match & program_running

                self._instructions[i_instruction](where=instruction_codes_match)

        return self._stack.get(-1, default=0.0)
