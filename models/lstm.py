# models/lstm.py

import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """
    Stronger LSTM model for time-series regression.
    - hidden_size 128
    - dropout reduced to 0.2 (we had underfitting)
    """
    def __init__(self, num_features, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.bn(last)
        return self.fc(last)
