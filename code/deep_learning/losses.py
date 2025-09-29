import torch
import torch.nn.functional as F

def coral_loss(logits, levels, importance_weights=None, reduction='mean'):
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

    term1 = (F.logsigmoid(logits) * levels
             + (F.logsigmoid(logits) - logits) * (1 - levels))

    if importance_weights is not None:
        term1 *= importance_weights.view(-1, 1)

    loss = -torch.sum(term1, dim=1)

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    elif reduction is not None:
        raise ValueError("Invalid value for `reduction`. Should be 'mean', "
                         "'sum', or None.")
    return loss


def levels_from_labelbatch(labels, num_classes, dtype=None):
    """
    Converts a batch of integer labels to extended binary levels.
    Source: https://github.com/Raschka-research-group/corn-ordinal-regression/blob/main/coral_pytorch/dataset.py

    Parameters
    ----------
    labels : torch.tensor, shape=(n_examples,)
        A batch of integer labels.

    num_classes : int
        The number of classes.

    dtype : torch.dtype, optional (default=None)
        The desired output data type.

    Returns
    ----------
    levels : torch.tensor, shape=(n_examples, num_classes-1)
        The labels in the extended binary format.

    """
    levels = []
    for label in labels:
        level = [1] * label.item() + [0] * (num_classes - 1 - label.item())
        levels.append(level)
    return torch.tensor(levels, dtype=dtype)
