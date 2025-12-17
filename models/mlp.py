# models/mlp.py

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """
    Simple baseline MLP for time series regression.
    Flattens the sequence and processes with fully connected layers.
    """
    def __init__(self, seq_len, num_features):
        super().__init__()
        input_dim = seq_len * num_features

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),  # Add BN to prevent overfitting
            nn.ReLU(),
            nn.Dropout(0.4),  # Increase dropout 0.2 -> 0.4

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout here too

            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, num_features]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, seq_len * num_features]
        return self.net(x)


class ImprovedMLP(nn.Module):
    """
    Improved MLP with Batch Normalization, deeper architecture,
    and better regularization.
    """
    def __init__(self, seq_len, num_features, dropout=0.3):
        super().__init__()
        input_dim = seq_len * num_features

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),  # Slightly less dropout

            # Layer 4
            nn.Linear(64, 32),
            nn.ReLU(),

            # Output
            nn.Linear(32, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for ReLU activation"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class ResidualMLP(nn.Module):
    """
    MLP with residual connections for better gradient flow.
    """
    def __init__(self, seq_len, num_features, dropout=0.3):
        super().__init__()
        input_dim = seq_len * num_features

        # Input projection
        self.input_proj = nn.Linear(input_dim, 256)

        # Residual blocks
        self.block1 = self._make_residual_block(256, 256, dropout)
        self.block2 = self._make_residual_block(256, 128, dropout)
        self.block3 = self._make_residual_block(128, 64, dropout)

        # Output
        self.output = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self._init_weights()

    def _make_residual_block(self, in_features, out_features, dropout):
        """Create a residual block"""
        return nn.ModuleDict({
            'linear1': nn.Linear(in_features, out_features),
            'bn1': nn.BatchNorm1d(out_features),
            'linear2': nn.Linear(out_features, out_features),
            'bn2': nn.BatchNorm1d(out_features),
            'dropout': nn.Dropout(dropout),
            'shortcut': nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        })

    def _residual_forward(self, x, block):
        """Forward pass through a residual block"""
        identity = block['shortcut'](x)

        out = block['linear1'](x)
        out = block['bn1'](out)
        out = torch.relu(out)
        out = block['dropout'](out)

        out = block['linear2'](out)
        out = block['bn2'](out)

        out += identity  # Residual connection
        out = torch.relu(out)

        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        # Input projection
        x = self.input_proj(x)
        x = torch.relu(x)

        # Residual blocks
        x = self._residual_forward(x, self.block1)
        x = self._residual_forward(x, self.block2)
        x = self._residual_forward(x, self.block3)

        # Output
        return self.output(x)