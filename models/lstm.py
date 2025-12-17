# models/lstm.py

import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """
    Basic LSTM predictor for time series regression.
    """
    def __init__(self, num_features, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        last_hidden = h_n[-1]  # [batch, hidden_size]

        return self.fc(last_hidden)


class ImprovedLSTM(nn.Module):
    """
    Improved LSTM with Bidirectional layers and Attention mechanism.
    """
    def __init__(self, num_features, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Important!
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Output layers with Batch Normalization
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),

            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM and Linear layers"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]

        # Attention mechanism
        attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Apply attention
        context = (lstm_out * attn_weights).sum(dim=1)  # [batch, hidden*2]

        # Output projection
        return self.fc(context)


class LSTM_CNN_Hybrid(nn.Module):
    """
    Hybrid model: CNN for local patterns + LSTM for temporal dependencies.
    """
    def __init__(self, num_features, hidden_size=128, dropout=0.3):
        super().__init__()

        # CNN for local feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # LSTM for sequential modeling
        self.lstm = nn.LSTM(
            64,
            hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, features]

        # CNN expects [batch, features, seq_len]
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # Back to [batch, seq_len', features']

        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)

        # Use last hidden state from both directions
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)

        return self.fc(hidden)