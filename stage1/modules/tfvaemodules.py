import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return x
