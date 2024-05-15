import torch


def dice_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    score = 2.0 * (y_pred * y_true).sum() / (y_pred.sum() + y_true.sum())
    return score
