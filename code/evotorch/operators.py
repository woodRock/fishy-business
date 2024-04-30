import torch

class AdditionalTorchFunctions:
    @staticmethod
    def _forbid_zero(x: torch.Tensor) -> torch.Tensor:
        """
        Move x away from 0 if its absolute value is less then 1e-4.
        """
        tolerance = 1e-4
        close_to_zero_from_pos = (x >= 0) & (x < tolerance)
        close_to_zero_from_neg = (x < 0) & (x > -tolerance)
        result = x.clone()
        result[close_to_zero_from_pos] = tolerance
        result[close_to_zero_from_neg] = -tolerance
        return result

    @classmethod
    def unary_div(cls, x: torch.Tensor) -> torch.Tensor:
        """
        Unary division with protection against division-by-zero.
        If x is not near zero, the result will be 1/x.
        If x is near zero, then it will first be moved away from zero.
        """
        return 1 / cls._forbid_zero(x)

    @classmethod
    def binary_div(cls, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Binary division with protection against division-by-zero.
        If b is not near zero, the result will be a/b.
        If b is near zero, then it will first be moved away from zero.
        """
        return a / cls._forbid_zero(b)
    
