# models/transformer.py

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """
    FIXED: Reduced overfitting with stronger regularization
    """
    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,  # Increased from 0.2
    ):
        super().__init__()

        # CNN with STRONGER regularization
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),

            nn.Conv1d(32, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout * 0.7),  # Added dropout here too
        )

        # Positional encoding with MORE dropout
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer with HIGHER dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,  # Increased
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )

        # Output with STRONGER regularization
        self.output_projection = nn.Sequential(
            nn.Dropout(dropout),  # Add dropout BEFORE first layer
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.BatchNorm1d(d_model // 2),  # Add BN
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape

        # CNN
        x_cnn = x.transpose(1, 2)
        x_cnn = self.cnn(x_cnn)
        x = x_cnn.transpose(1, 2)

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer_encoder(x)

        # Attention pooling
        attn_weights = self.attention_pool(x)
        x = (x * attn_weights).sum(dim=1)

        # Output
        output = self.output_projection(x)

        return output


class HybridTransformer(nn.Module):
    """
    FIXED: Simplified hybrid to reduce overfitting
    """
    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,  # Reduced from 2
        lstm_hidden: int = 64,
        dropout: float = 0.4,  # Increased from 0.2
    ):
        super().__init__()

        # Simpler input projection
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # LIGHTER Transformer (1 layer only)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # Reduced
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Simpler LSTM
        self.lstm = nn.LSTM(
            d_model,
            lstm_hidden,
            num_layers=1,  # Single layer
            batch_first=True,
            dropout=0,  # No dropout inside LSTM
            bidirectional=False  # Unidirectional only
        )

        # Output with strong regularization
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.BatchNorm1d(lstm_hidden // 2),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1)
        )

    def forward(self, x):
        # Project
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer(x)

        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)

        # Use final hidden state
        hidden = h_n[-1]

        # Output
        return self.fc(hidden)