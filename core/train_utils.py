import torch


def parse_device():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return DEVICE
