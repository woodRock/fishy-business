import torch
from typing import Callable, Optional
from collections import namedtuple
from evotorch.tools.structures import CList

_PopResult = namedtuple("_PopResult", ["tensor_a", "pop_mask"])
_PopPairResult = namedtuple("_PopPairResult", ["tensor_a", "tensor_b", "pop_mask"])

class Instruction:
    def __init__(
        self,
        *,
        inputs: torch.Tensor,
        stack: CList,
        arity: int,
        function: Optional[Callable] = None,
        input_slot: Optional[int] = None,
        operation: Optional[str] = None,
    ):
        [batch_size] = stack.batch_shape
        inputs_batch_size, input_size = inputs.shape
        assert inputs_batch_size == batch_size

        self.input_size = input_size
        self.stack = stack
        self.inputs = inputs
        self.arity = int(arity)

        self.function = None
        self.input_slot = None
        self.operation = None

        instr_definitions = 0

        if function is not None:
            self.function = function
            instr_definitions += 1

        if input_slot is not None:
            assert self.arity == 0
            self.input_slot = input_slot
            instr_definitions += 1

        if operation is not None:
            assert self.arity == 0
            assert operation in ("pass", "swap", "duplicate")
            self.operation = operation
            instr_definitions += 1

        assert instr_definitions == 1, "Please specify only one of these: `function`, `input_slot`, or `operation`."        
        assert self.arity in (0, 1, 2)

    def _pop(self, where: torch.Tensor) -> _PopResult:
        suitable = self.stack.length >= 1
        where = where & suitable
        return _PopResult(tensor_a=self.stack.pop_(where=where), pop_mask=where)

    def _pop_pair(self, where: torch.Tensor) -> _PopPairResult:
        suitable = self.stack.length >= 2
        where = where & suitable
        b = self.stack.pop_(where=where)
        a = self.stack.pop_(where=where)
        return _PopPairResult(tensor_a=a, tensor_b=b, pop_mask=where)

    def _push(self, x: torch.Tensor, where: torch.Tensor):
        self.stack.push_(x, where=where)

    def _push_input(self, input_slot: int, where: torch.Tensor):
        input_values = self.inputs[:, input_slot]
        self.stack.push_(input_values, where=where)

    def __call__(self, where: torch.Tensor):
        if self.function is not None:
            fn = self.function
            arity = self.arity

            if arity == 0:
                self._push(fn(), where=where)
            elif arity == 1:
                a, where = self._pop(where=where)
                self._push(fn(a), where=where)
            elif arity == 2:
                a, b, where = self._pop_pair(where=where)
                self._push(fn(a, b), where=where)
            else:
                assert False

        if self.input_slot is not None:
            self._push_input(self.input_slot, where=where)

        if self.operation is not None:
            if self.operation == "pass":
                pass
            elif self.operation == "swap":
                a, b, where = self._pop_pair(where=where)
                self._push(b, where=where)
                self._push(a, where=where)
            elif self.operation == "duplicate":
                a, where = self._pop(where=where)
                self._push(a, where=where)
                self._push(a, where=where)
            else:
                assert False, f"unknown operation: {self.operation}"

    def __repr__(self) -> str:
        result = []

        def puts(*xs: str):
            for x in xs:
                result.append(str(x))

        puts(type(self).__name__, "(")

        if self.function is not None:
            if hasattr(self.function, "__name__"):
                fn_name = self.function.__name__
            else:
                fn_name = repr(self.function)
            puts("function=", fn_name)

        if self.input_slot is not None:
            puts("input_slot=", self.input_slot)

        if self.operation is not None:
            puts("operation=", repr(self.operation))

        puts(", arity=", self.arity)
        puts(")")
        return "".join(result)
