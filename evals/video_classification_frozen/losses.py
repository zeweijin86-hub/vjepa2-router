from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss

_VALID_REDUCTIONS = ("none", "mean", "sum")


class FocalLoss(_WeightedLoss):
    """Multiclass Focal Loss using numerically stable FP32 log_softmax.

    Computes ``-weight_t * (1 - p_t)^gamma * log(p_t)`` for each sample.
    Logits are promoted to ``float32`` before computing probabilities to avoid
    underflow or NaN gradients under FP16 autocast.

    Args:
        gamma (float): Focusing parameter that down-weights easy examples. Default: 2.0.
        weight (Tensor, optional): Manual rescaling weight tensor of shape ``(C,)``.
        reduction (str): Reduction mode: ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``.
    """

    __constants__ = ["gamma", "reduction"]
    gamma: float

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        if reduction not in _VALID_REDUCTIONS:
            raise ValueError(f"Invalid reduction '{reduction}'; expected one of {_VALID_REDUCTIONS}.")
        super().__init__(weight=weight, reduction=reduction)
        self.gamma = gamma

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        logits = input.float()
        log_softmax = F.log_softmax(logits, dim=1)
        softmax = torch.exp(log_softmax)

        target = target.long()
        log_p = log_softmax.gather(1, target.unsqueeze(1)).squeeze(1)
        p = softmax.gather(1, target.unsqueeze(1)).squeeze(1)

        loss = -log_p * (1.0 - p).clamp(min=0.0) ** self.gamma

        if self.weight is not None:
            weight = self.weight.to(device=logits.device, dtype=logits.dtype).gather(0, target)
            loss = loss * weight

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
