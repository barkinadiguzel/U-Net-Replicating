import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target, weight_map=None):
        """
        logits: BxCxHxW (raw outputs)
        target: BxHxW (class indices)
        weight_map: BxHxW (optional)
        """
        loss = F.cross_entropy(logits, target, reduction='none')
        if weight_map is not None:
            loss = loss * weight_map
        return loss.mean()
