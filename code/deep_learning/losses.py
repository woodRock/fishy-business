import torch
import torch.nn.functional as F


def coral_loss(logits, levels, importance_weights=None, reduction="mean"):
    """Computes the CORAL loss.
    Source: https://github.com/Raschka-research-group/corn-ordinal-regression/blob/main/coral_pytorch/losses.py

    Parameters
    ----------
    logits : torch.tensor, shape=(n_examples, n_classes-1)
        Outputs of the CORAL layer.

    levels : torch.tensor, shape=(n_examples, n_classes-1)
        True labels represented as extended binary vectors
        (via `levels_from_labelbatch`).

    importance_weights : torch.tensor, shape=(n_examples,), optional (default=None)
        Optional weights for the examples in the batch.
        If None, all examples are weighted equally.

    reduction : str or None (default='mean')
        If 'mean' or 'sum', returns the averaged or summed loss across
        all examples. If None, returns a tensor of shape (n_examples,)
        with the loss for each example.

    Returns
    ----------
    loss : torch.tensor
        A torch.tensor containing a single loss value (if `reduction='mean'` or
        `reduction='sum'`) or a loss value for each example (`reduction=None`).

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


def levels_from_labelbatch(labels, num_classes, dtype=None):
    """
    Converts a batch of integer labels to extended binary levels for CORAL.
    Vectorized implementation.
    """
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
        
    # Ensure labels are long integers for comparison
    labels = labels.long()

    # Create a tensor of shape (batch_size, num_classes - 1) with values 0, 1, 2, ...
    rank = torch.arange(num_classes - 1, device=labels.device, dtype=labels.dtype).expand(labels.size(0), -1)

    # Create a mask where rank < labels
    levels = (rank < labels.unsqueeze(1)).to(dtype=dtype)

    return levels


def cumulative_link_loss(logits, labels, num_classes, reduction="mean"):
    """
    Computes the Cumulative Link loss.

    This loss treats ordinal regression as K-1 binary classification problems.
    For each class boundary, it predicts whether the true label is beyond that boundary.

    Parameters
    ----------
    logits : torch.tensor, shape=(n_examples, n_classes-1)
        Outputs from the model.
    labels : torch.tensor, shape=(n_examples,)
        True integer labels.
    num_classes : int
        The total number of ordinal classes.
    reduction : str or None (default='mean')
        Type of reduction to apply to the loss.

    Returns
    ----------
    loss : torch.tensor
        The computed loss value.
    """
    # Convert integer labels to cumulative binary format, e.g., for 5 classes:
    # 0 -> [0, 0, 0, 0]
    # 1 -> [1, 0, 0, 0]
    # 2 -> [1, 1, 0, 0]
    # 3 -> [1, 1, 1, 0]
    # 4 -> [1, 1, 1, 1]
    cumulative_labels = levels_from_labelbatch(labels, num_classes, dtype=logits.dtype).to(logits.device)

    if not logits.shape == cumulative_labels.shape:
        raise ValueError("Shape of logits and cumulative_labels must be the same.")

    return F.binary_cross_entropy_with_logits(logits, cumulative_labels, reduction=reduction)