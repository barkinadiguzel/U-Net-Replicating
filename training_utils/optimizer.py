import torch

def get_optimizer(model, lr=0.01, momentum=0.99):
    """
    Returns SGD optimizer with momentum
    """
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
