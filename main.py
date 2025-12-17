# main.py
import os
import torch
import pandas as pd

from utils.data import load_time_series_dataloaders
from utils.train import train_model, evaluate, regression_metrics

from models.mlp import BaselineMLP
from models.lstm import LSTMPredictor
from models.transformer import TimeSeriesTransformer, HybridTransformer
# Assuming you have these plot functions
# from utils.plots import plot_loss_curves, plot_true_vs_pred, plot_scatter


# ======================
# Configuration
# ======================
CSV_PATH = "dc_extended.csv"
WINDOW_SIZE = 90
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 150
PATIENCE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ======================
# Load Data
# ======================
print("\n=== Loading Data ===")
data = load_time_series_dataloaders(
    csv_path=CSV_PATH,
    target_col="close_USD",
    window_size=WINDOW_SIZE,
    batch_size=BATCH_SIZE,
)

train_loader = data["train_loader"]
val_loader = data["val_loader"]
test_loader = data["test_loader"]
target_scaler = data["target_scaler"]  # IMPORTANT!
seq_len = data["seq_len"]
num_features = data["num_features"]

print(f"Sequence length: {seq_len}")
print(f"Number of features: {num_features}")
print(f"Train samples: {len(train_loader.dataset)}")
print(f"Val samples: {len(val_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")


# ======================
# 1. Baseline MLP
# ======================
print("\n" + "="*60)
print("=== Training Baseline MLP ===")
print("="*60)
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

mlp_test_loss, mlp_pred_scaled, mlp_true_scaled = evaluate(
    mlp, test_loader, torch.nn.MSELoss(), DEVICE
)

# INVERSE TRANSFORM to original scale
mlp_pred = target_scaler.inverse_transform(mlp_pred_scaled.reshape(-1, 1)).flatten()
mlp_true = target_scaler.inverse_transform(mlp_true_scaled.reshape(-1, 1)).flatten()

mlp_mse, mlp_rmse, mlp_mae, mlp_mape, mlp_r2 = regression_metrics(mlp_true, mlp_pred)
print(f"\nMLP Results:")
print(f"  RMSE: ${mlp_rmse:,.2f}")
print(f"  MAE:  ${mlp_mae:,.2f}")
print(f"  MAPE: {mlp_mape:.2f}%")
print(f"  R²:   {mlp_r2:.4f}")


# ======================
# 2. LSTM Model
# ======================
print("\n" + "="*60)
print("=== Training LSTM Model ===")
print("="*60)
lstm = LSTMPredictor(num_features=num_features, hidden_size=128, num_layers=2)

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

lstm_test_loss, lstm_pred_scaled, lstm_true_scaled = evaluate(
    lstm, test_loader, torch.nn.MSELoss(), DEVICE
)

# INVERSE TRANSFORM
lstm_pred = target_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
lstm_true = target_scaler.inverse_transform(lstm_true_scaled.reshape(-1, 1)).flatten()

lstm_mse, lstm_rmse, lstm_mae, lstm_mape, lstm_r2 = regression_metrics(lstm_true, lstm_pred)
print(f"\nLSTM Results:")
print(f"  RMSE: ${lstm_rmse:,.2f}")
print(f"  MAE:  ${lstm_mae:,.2f}")
print(f"  MAPE: {lstm_mape:.2f}%")
print(f"  R²:   {lstm_r2:.4f}")


# ======================
# 3. Transformer Model (Lightweight)
# ======================
print("\n" + "="*60)
print("=== Training Transformer Model (Lightweight) ===")
print("="*60)
transformer = TimeSeriesTransformer(
    num_features=num_features,
    d_model=64,     # Smaller model for small dataset
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.2
)

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

trf_test_loss, trf_pred_scaled, trf_true_scaled = evaluate(
    transformer, test_loader, torch.nn.MSELoss(), DEVICE
)

# INVERSE TRANSFORM
trf_pred = target_scaler.inverse_transform(trf_pred_scaled.reshape(-1, 1)).flatten()
trf_true = target_scaler.inverse_transform(trf_true_scaled.reshape(-1, 1)).flatten()

trf_mse, trf_rmse, trf_mae, trf_mape, trf_r2 = regression_metrics(trf_true, trf_pred)
print(f"\nTransformer Results:")
print(f"  RMSE: ${trf_rmse:,.2f}")
print(f"  MAE:  ${trf_mae:,.2f}")
print(f"  MAPE: {trf_mape:.2f}%")
print(f"  R²:   {trf_r2:.4f}")


# ======================
# 4. Hybrid Transformer-LSTM Model
# ======================
print("\n" + "="*60)
print("=== Training Hybrid Transformer-LSTM Model ===")
print("="*60)
hybrid = HybridTransformer(
    num_features=num_features,
    d_model=64,
    nhead=4,
    num_layers=1,  # Reduced
    lstm_hidden=64,
    dropout=0.4  # Increased
)

hybrid, hyb_tr, hyb_val = train_model(
    hybrid,
    train_loader,
    val_loader,
    epochs=EPOCHS,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE,
    name="hybrid",
    device=DEVICE,
)

hyb_test_loss, hyb_pred_scaled, hyb_true_scaled = evaluate(
    hybrid, test_loader, torch.nn.MSELoss(), DEVICE
)

# INVERSE TRANSFORM
hyb_pred = target_scaler.inverse_transform(hyb_pred_scaled.reshape(-1, 1)).flatten()
hyb_true = target_scaler.inverse_transform(hyb_true_scaled.reshape(-1, 1)).flatten()

hyb_mse, hyb_rmse, hyb_mae, hyb_mape, hyb_r2 = regression_metrics(hyb_true, hyb_pred)
print(f"\nHybrid Results:")
print(f"  RMSE: ${hyb_rmse:,.2f}")
print(f"  MAE:  ${hyb_mae:,.2f}")
print(f"  MAPE: {hyb_mape:.2f}%")
print(f"  R²:   {hyb_r2:.4f}")


# ======================
# Summary Table
# ======================
print("\n" + "="*60)
print("=== FINAL MODEL COMPARISON ===")
print("="*60)
print(f"{'Model':<20} {'RMSE ($)':<15} {'MAE ($)':<15} {'MAPE (%)':<12} {'R²':<10}")
print("-" * 72)
print(f"{'MLP':<20} {mlp_rmse:>13,.2f}  {mlp_mae:>13,.2f}  {mlp_mape:>10.2f}  {mlp_r2:>8.4f}")
print(f"{'LSTM':<20} {lstm_rmse:>13,.2f}  {lstm_mae:>13,.2f}  {lstm_mape:>10.2f}  {lstm_r2:>8.4f}")
print(f"{'Transformer':<20} {trf_rmse:>13,.2f}  {trf_mae:>13,.2f}  {trf_mape:>10.2f}  {trf_r2:>8.4f}")
print(f"{'Hybrid Trans-LSTM':<20} {hyb_rmse:>13,.2f}  {hyb_mae:>13,.2f}  {hyb_mape:>10.2f}  {hyb_r2:>8.4f}")


# ======================
# Save Results
# ======================
results = {
    'MODEL': ['MLP', 'LSTM', 'Transformer', 'Hybrid'],
    'RMSE': [mlp_rmse, lstm_rmse, trf_rmse, hyb_rmse],
    'MAE': [mlp_mae, lstm_mae, trf_mae, hyb_mae],
    'MAPE': [mlp_mape, lstm_mape, trf_mape, hyb_mape],
    'R2': [mlp_r2, lstm_r2, trf_r2, hyb_r2]
}

results_df = pd.DataFrame(results)
results_df.to_csv('results_table.csv', index=False)
print("\n✓ Results saved to results_table.csv")

# Find best model
best_idx = results_df['RMSE'].idxmin()
best_model_name = results_df.loc[best_idx, 'MODEL']
best_rmse = results_df.loc[best_idx, 'RMSE']
print(f"\n✓ Best model: {best_model_name} (RMSE: ${best_rmse:,.2f})")

# Uncomment if you have plotting functions
# ======================
# Plots
# ======================
# os.makedirs("plots", exist_ok=True)
#
# plot_loss_curves(
#     mlp_tr, mlp_val,
#     lstm_tr, lstm_val,
#     trf_tr, trf_val,
#     out_dir="plots"
# )
#
# # Plot best model (LSTM typically)
# plot_true_vs_pred(
#     lstm_true,
#     lstm_pred,
#     title="LSTM – True vs Predicted (Test)",
#     out_path="plots/lstm_true_vs_pred.png"
# )
#
# plot_scatter(
#     lstm_true,
#     lstm_pred,
#     title="LSTM – True vs Predicted Scatter (Test)",
#     out_path="plots/lstm_scatter.png"
# )

print("\n" + "="*60)
print("Training complete!")
print("="*60)