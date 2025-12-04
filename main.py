# main.py
# MLP, LSTM, and Transformer models
import os
import torch
import pandas as pd

from utils.data import load_time_series_dataloaders
from utils.train import train_model, evaluate, regression_metrics

from models.mlp import BaselineMLP
from models.lstm import LSTMPredictor
from models.transformer import TimeSeriesTransformer
from utils.plots import plot_loss_curves, plot_true_vs_pred, plot_scatter


# ======================
# Configuration
# ======================
CSV_PATH = "dc_extended.csv"
WINDOW_SIZE = 90
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 150
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# Load Data
# ======================
data = load_time_series_dataloaders(
    csv_path=CSV_PATH,
    target_col="close_USD",
    window_size=WINDOW_SIZE,
    batch_size=BATCH_SIZE,
)

train_loader = data["train_loader"]
val_loader = data["val_loader"]
test_loader = data["test_loader"]
seq_len = data["seq_len"]
num_features = data["num_features"]


# ======================
# 1. Baseline MLP
# ======================
print("\n=== Training Baseline MLP ===")
mlp = BaselineMLP(seq_len, num_features)

mlp, mlp_tr, mlp_val = train_model(
    mlp,
    train_loader,
    val_loader,
    epochs=EPOCHS,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE,
    name="mlp",
    device=DEVICE,
)

mlp_test_loss, mlp_pred, mlp_true = evaluate(
    mlp, test_loader, torch.nn.MSELoss(), DEVICE
)

mlp_mse, mlp_rmse, mlp_mae = regression_metrics(mlp_true, mlp_pred)
print(f"MLP Test MSE: {mlp_mse:.4f} | RMSE: {mlp_rmse:.4f} | MAE: {mlp_mae:.4f}")


# ======================
# 2. LSTM Model
# ======================
print("\n=== Training LSTM Model ===")
lstm = LSTMPredictor(num_features=num_features)

lstm, lstm_tr, lstm_val = train_model(
    lstm,
    train_loader,
    val_loader,
    epochs=EPOCHS,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE,
    name="lstm",
    device=DEVICE,
)

lstm_test_loss, lstm_pred, lstm_true = evaluate(
    lstm, test_loader, torch.nn.MSELoss(), DEVICE
)

lstm_mse, lstm_rmse, lstm_mae = regression_metrics(lstm_true, lstm_pred)
print(f"LSTM Test MSE: {lstm_mse:.4f} | RMSE: {lstm_rmse:.4f} | MAE: {lstm_mae:.4f}")


# ======================
# 3. Transformer Model
# ======================
print("\n=== Training Transformer Model ===")
transformer = TimeSeriesTransformer(num_features=num_features)

transformer, trf_tr, trf_val = train_model(
    transformer,
    train_loader,
    val_loader,
    epochs=EPOCHS,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE,
    name="transformer",
    device=DEVICE,
)

trf_test_loss, trf_pred, trf_true = evaluate(
    transformer, test_loader, torch.nn.MSELoss(), DEVICE
)

trf_mse, trf_rmse, trf_mae = regression_metrics(trf_true, trf_pred)
print(f"Transformer Test MSE: {trf_mse:.4f} | RMSE: {trf_rmse:.4f} | MAE: {trf_mae:.4f}")


# ======================
# Summary Table
# ======================
print("\n=== Final Model Comparison ===")
print(f"MLP        → RMSE: {mlp_rmse:.4f} | MAE: {mlp_mae:.4f}")
print(f"LSTM       → RMSE: {lstm_rmse:.4f} | MAE: {lstm_mae:.4f}")
print(f"Transformer→ RMSE: {trf_rmse:.4f} | MAE: {trf_mae:.4f}")

# ======================
# Plots
# ======================
plot_loss_curves(
    mlp_tr, mlp_val,
    lstm_tr, lstm_val,
    trf_tr, trf_val,
    out_dir="plots"
)

# LSTM plots (best model)
plot_true_vs_pred(
    lstm_true,
    lstm_pred,
    title="LSTM – True vs Predicted (Test)",
    out_path="plots/lstm_true_vs_pred.png"
)

plot_scatter(
    lstm_true,
    lstm_pred,
    title="LSTM – True vs Predicted Scatter (Test)",
    out_path="plots/lstm_scatter.png"
)

# Transformer plots (optional)
plot_true_vs_pred(
    trf_true,
    trf_pred,
    title="Transformer – True vs Predicted (Test)",
    out_path="plots/transformer_true_vs_pred.png"
)

plot_scatter(
    trf_true,
    trf_pred,
    title="Transformer – True vs Predicted Scatter (Test)",
    out_path="plots/transformer_scatter.png"
)

# ======================
# Summary Table (numeric)
# ======================

results = {
    'MODEL':    ['MLP', 'LSTM', 'Transformer'],
    'MSE'  :    [mlp_mse, lstm_mse, trf_mse],
    'RMSE' :    [mlp_rmse, lstm_rmse, trf_rmse],
    'MAE'  :    [mlp_mae, lstm_mae, trf_mae]
}

results_df = pd.DataFrame(results)
print('\n === Regression result table === ')
print(results_df)


results_df.to_csv('results.table.csv', index=False)