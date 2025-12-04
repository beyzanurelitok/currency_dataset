import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    def __init__(self, seq_len, num_features):
        super().__init__()
        input_dim = seq_len * num_features
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
