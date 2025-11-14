import torch
import torch.nn as nn

def init_weights_he(module):
    """
    Initialize weights according to He initialization.
    Applies only to Conv and Linear layers.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


