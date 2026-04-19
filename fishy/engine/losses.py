import torch
import torch.nn.functional as F
from typing import Optional


def coral_loss(
    logits: torch.Tensor,
    levels: torch.Tensor,
    importance_weights: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Computes the CORAL (Consistent Rank Logits) loss for ordinal regression.

    Source: https://github.com/Raschka-research-group/corn-ordinal-regression/blob/main/coral_pytorch/losses.py

    Args:
        logits (torch.Tensor): Outputs of the CORAL layer, shape (n_examples, n_classes-1).
        levels (torch.Tensor): True labels represented as extended binary vectors (via `levels_from_labelbatch`), shape (n_examples, n_classes-1).
        importance_weights (Optional[torch.Tensor], optional): Weights for the examples in the batch. Defaults to None.
        reduction (str, optional): Reduction to apply to the loss ('mean', 'sum', or None). Defaults to "mean".

    Returns:
        torch.Tensor: The computed loss value (scalar if reduction is not None, else vector).

    Raises:
        ValueError: If logits and levels shapes do not match.
        ValueError: If reduction is not one of 'mean', 'sum', or None.

    Examples:
        >>> import torch
        >>> logits = torch.tensor([[10.0, 5.0], [-5.0, -10.0]])
        >>> levels = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        >>> loss = coral_loss(logits, levels)
        >>> float(loss) < 0.1
        True
        >>> loss_sum = coral_loss(logits, levels, reduction='sum')
        >>> float(loss_sum) > float(loss)
        True
    """
    if not logits.shape == levels.shape:
        raise ValueError("Please provide logits and levels of equal shape.")

    term1 = F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (
        1 - levels
    )

    if importance_weights is not None:
        term1 *= importance_weights.view(-1, 1)

    loss = -torch.sum(term1, dim=1)

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction is not None:
        raise ValueError(
            "Invalid value for `reduction`. Should be 'mean', " "'sum', or None."
        )
    return loss


def levels_from_labelbatch(
    labels: torch.Tensor, num_classes: int, dtype: torch.dtype = None
) -> torch.Tensor:
    """
    Converts a batch of integer labels to extended binary levels for CORAL.

    For example, with 5 classes, label 2 becomes [1, 1, 0, 0].
    Vectorized implementation.

    Args:
        labels (torch.Tensor): Batch of integer labels.
        num_classes (int): Total number of ordinal classes.
        dtype (torch.dtype, optional): Desired dtype of the output tensor. Defaults to None.

    Returns:
        torch.Tensor: Binary levels tensor of shape (batch_size, num_classes - 1).

    Examples:
        >>> import torch
        >>> labels = torch.tensor([0, 1, 2])
        >>> levels = levels_from_labelbatch(labels, num_classes=4, dtype=torch.float32)
        >>> levels
        tensor([[0., 0., 0.],
                [1., 0., 0.],
                [1., 1., 0.]])
    """
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    # Ensure labels are long integers for comparison
    labels = labels.long()

    # Create a tensor of shape (batch_size, num_classes - 1) with values 0, 1, 2, ...
    rank = torch.arange(
        num_classes - 1, device=labels.device, dtype=labels.dtype
    ).expand(labels.size(0), -1)

    # Create a mask where rank < labels
    levels = (rank < labels.unsqueeze(1)).to(dtype=dtype)

    return levels


def cumulative_link_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Computes the Cumulative Link loss for ordinal regression.

    This loss treats ordinal regression as K-1 binary classification problems.
    For each class boundary, it predicts whether the true label is beyond that boundary.

    Args:
        logits (torch.Tensor): Model outputs, shape (n_examples, n_classes-1).
        labels (torch.Tensor): True integer labels, shape (n_examples,).
        num_classes (int): The total number of ordinal classes.
        reduction (str, optional): Type of reduction to apply ('mean', 'sum', 'none'). Defaults to "mean".

    Returns:
        torch.Tensor: The computed loss value.

    Raises:
        ValueError: If shape mismatch between logits and derived cumulative labels.

    Examples:
        >>> import torch
        >>> logits = torch.tensor([[10.0, 10.0], [-10.0, -10.0]])
        >>> labels = torch.tensor([2, 0])
        >>> loss = cumulative_link_loss(logits, labels, num_classes=3)
        >>> float(loss) < 0.1
        True
    """
    # Convert integer labels to cumulative binary format, e.g., for 5 classes:
    # 0 -> [0, 0, 0, 0]
    # 1 -> [1, 0, 0, 0]
    # 2 -> [1, 1, 0, 0]
    # 3 -> [1, 1, 1, 0]
    # 4 -> [1, 1, 1, 1]
    cumulative_labels = levels_from_labelbatch(
        labels, num_classes, dtype=logits.dtype
    ).to(logits.device)

    if not logits.shape == cumulative_labels.shape:
        raise ValueError("Shape of logits and cumulative_labels must be the same.")

    return F.binary_cross_entropy_with_logits(
        logits, cumulative_labels, reduction=reduction
    )


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for imbalanced classification.
    Reduces the loss for well-classified examples and focuses on hard examples.
    
    Source: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
