import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, input_size=32):
        super().__init__()
        self.input_size=input_size

    def forward(self, x):
        # print(x)
        # exit()
        x =x.type(torch.long)
        return x