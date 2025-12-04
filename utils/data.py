# utils/data.py

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic technical indicators based on close_USD.
    - daily return
    # - log return
    - EMA(7), EMA(21)
    - RSI(14)
    """
    
    # --- FIX START: Ensure numeric types ---
    # We force conversion to numeric, turning non-parseable errors into NaNs
    cols_to_fix = ['close_USD', 'open_USD', 'high_USD', 'low_USD', 'volume']
    for col in cols_to_fix:
        if col in df.columns:
            # remove commas if they exist as strings, then convert
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    

    close = df["close_USD"]

    # Simple returns and log-returns
    df["return_1d"] = close.pct_change()
    df["log_return_1d"] = np.log(close).diff()

    # Exponential moving averages
    df["ema_7"] = close.ewm(span=7, adjust=False).mean()
    df["ema_21"] = close.ewm(span=21, adjust=False).mean()

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    period = 14
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Handle division by zero for RSI
    rs = avg_gain / avg_loss.replace(0, np.nan) 
    df["rsi_14"] = 100 - (100 / (1 + rs))
    
    # Fill RSI NaNs that might occur at the start with 50 or forward fill
    df["rsi_14"] = df["rsi_14"].fillna(50)

    # Drop initial NaNs created by indicators (EMA/Returns)
    df = df.dropna().reset_index(drop=True)
    return df


def create_sequences(df: pd.DataFrame, feature_cols, target_col: str, window: int):
    """
    Convert time-series DataFrame into supervised sequences.
    X: (samples, window, num_features)
    y: (samples,)
    """
    values = df[feature_cols].values.astype(np.float32)
    t_index = feature_cols.index(target_col)

    X_list, y_list = [], []
    for i in range(len(df) - window):
        x = values[i:i + window]
        y = values[i + window, t_index]
        X_list.append(x)
        y_list.append(y)

    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y


def load_time_series_dataloaders(
    csv_path: str,
    target_col: str = "close_USD",
    window_size: int = 90,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    # ----- Load raw CSV -----
    # Added thousands=',' to handle standard currency formatting automatically if present
    df = pd.read_csv(csv_path, thousands=',')

    # Rename first column to 'date' if it looks like a timestamp
    if df.columns[0].lower() in ["date", "time", "timestamp", "datetime"] or df.columns[0].startswith("Unnamed"):
        df.rename(columns={df.columns[0]: "date"}, inplace=True)

    # Sort by date if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    # ----- Keep only USD-based price columns + volume -----
    # We ignore SAR columns to reduce noise.
    keep_cols = ["date", "open_USD", "high_USD", "low_USD", "close_USD", "volume"]
    
    # Filter only columns that actually exist in the CSV to avoid KeyError
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # ----- Add technical indicators -----
    # This function now handles the type conversion
    df = _add_technical_indicators(df)

    # Feature columns: USD + indicators
    feature_cols = [
        "open_USD",
        "high_USD",
        "low_USD",
        "close_USD",
        "volume",
        "return_1d",
        "log_return_1d",
        "ema_7",
        "ema_21",
        "rsi_14",
    ]
    
    # Ensure we only define features that exist (in case volume was missing, etc.)
    feature_cols = [c for c in feature_cols if c in df.columns]

    assert target_col in feature_cols, f"{target_col} not in feature columns: {feature_cols}"

    # ----- Create sequences -----
    X_all, y_all = create_sequences(df, feature_cols, target_col, window_size)
    num_samples, seq_len, num_features = X_all.shape

    # ----- Time-based split -----
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size

    X_train = X_all[:train_size]
    y_train = y_all[:train_size]

    X_val = X_all[train_size:train_size + val_size]
    y_val = y_all[train_size:train_size + val_size]

    X_test = X_all[train_size + val_size:]
    y_test = y_all[train_size + val_size:]

    # ----- Feature scaling (fit on train only) -----
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, num_features)
    scaler.fit(X_train_flat)

    def scale_X(X):
        X_flat = X.reshape(-1, num_features)
        X_scaled = scaler.transform(X_flat)
        return X_scaled.reshape(-1, seq_len, num_features)

    X_train = scale_X(X_train)
    X_val = scale_X(X_val)
    X_test = scale_X(X_test)

    # ----- Datasets & loaders -----
    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)
    test_ds = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler": scaler,
        "seq_len": seq_len,
        "num_features": num_features,
        "feature_cols": feature_cols,
    }