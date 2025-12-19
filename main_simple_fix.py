"""
Improved Training Strategy - Son Optimizasyon
-------------------------------------------------
Hocaların önerileri doğrultusunda:
1. ✅ Zamansal sıralı split (zaten yapılıyor)
2. ✅ Train+Val birleştirme (zaten yapılıyor)
3. ✅ Target normalization (zaten yapılıyor)

Yeni iyileştirmeler:
4. Fixed random seed → Reproducibility
5. Simplified models → Less overfitting
6. Better retrain strategy → best_epoch kadar retrain (1.5x değil)
7. Multiple runs → Sonuçların ortalamasını al
8. Learning rate warmup → Daha stabil eğitim
"""

import os
import torch
import pandas as pd
import numpy as np
import random

from utils.data import load_time_series_dataloaders
from utils.train import train_model, evaluate, regression_metrics, retrain_on_train_val
from models.mlp import ImprovedMLP
from models.lstm import ImprovedLSTM
from models.transformer import TimeSeriesTransformer


def set_seed(seed=42):
    """Reproducibility için seed sabitle"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================
# Configuration
# ======================
CSV_PATH = "dc_extended.csv"
WINDOW_SIZE = 90
BATCH_SIZE = 64
EPOCHS = 150  # 200'den düşürdük
PATIENCE = 25  # 30'dan düşürdük
NUM_RUNS = 3  # Her modeli 3 kez çalıştır
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print("\n" + "="*70)
print("FİNAL OPTİMİZE EĞİTİM STRATEJİSİ")
print("="*70)
print(f"✓ {NUM_RUNS} run ile ensemble sonuç alınacak")
print(f"✓ Reproducibility için seed sabitlendi")

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
target_scaler = data["target_scaler"]
seq_len = data["seq_len"]
num_features = data["num_features"]

print(f"✓ Sequence length: {seq_len}")
print(f"✓ Number of features: {num_features}")
print(f"✓ Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")


# ======================
# Simplified Model Configurations
# ======================
model_configs = {
    "MLP": {
        "class": ImprovedMLP,
        "kwargs": {
            "seq_len": seq_len,
            "num_features": num_features,
            "dropout": 0.25  # 0.3'ten düşürdük
        },
        "lr": 1e-3,
        "weight_decay": 5e-6,  # Daha az regularization
    },

    "LSTM": {
        "class": ImprovedLSTM,
        "kwargs": {
            "num_features": num_features,
            "hidden_size": 96,  # 128'den küçülttük
            "num_layers": 2,  # 3'ten düşürdük
            "dropout": 0.25
        },
        "lr": 1e-3,
        "weight_decay": 5e-6,
    },

    "Transformer": {
        "class": TimeSeriesTransformer,
        "kwargs": {
            "num_features": num_features,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 192,  # 256'dan küçülttük
            "dropout": 0.25,
        },
        "lr": 8e-4,  # Biraz daha düşük
        "weight_decay": 1e-5,
    },
}


# ======================
# Training Loop with Multiple Runs
# ======================
all_results = []

for model_name, config in model_configs.items():
    print("\n" + "="*70)
    print(f"=== {model_name.upper()} ===")
    print("="*70)

    run_results = []

    for run in range(NUM_RUNS):
        print(f"\n--- Run {run + 1}/{NUM_RUNS} ---")

        # Her run için farklı seed
        set_seed(42 + run)

        # PHASE 1: Train + Val ayrı
        print(f"[PHASE 1] Train ve Val ile eğitim...")

        model = config["class"](**config["kwargs"])

        model_trained, train_losses, val_losses, best_epoch = train_model(
            model,
            train_loader,
            val_loader,
            epochs=EPOCHS,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            patience=PATIENCE,
            name=f"{model_name.lower()}_phase1_run{run}",
            device=DEVICE,
            scheduler_type="plateau",
            return_best_epoch=True,
        )

        print(f"✓ Best epoch: {best_epoch}")

        # CRITICAL FIX: Retrain için best_epoch kadar epoch (1.5x değil!)
        # Çünkü combined data zaten daha fazla sample içeriyor
        retrain_epochs = best_epoch
        retrain_epochs = max(retrain_epochs, 15)  # Min 15
        retrain_epochs = min(retrain_epochs, 60)  # Max 60

        print(f"→ Retrain: {retrain_epochs} epoch (best_epoch)")

        # PHASE 2: Train+Val birleştir
        print(f"[PHASE 2] Train+Val birleştirilerek {retrain_epochs} epoch eğitiliyor...")

        model_final = config["class"](**config["kwargs"])

        model_final = retrain_on_train_val(
            model_final,
            train_loader,
            val_loader,
            epochs=retrain_epochs,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            name=f"{model_name.lower()}_final_run{run}",
            device=DEVICE,
        )

        # PHASE 3: Test tahminleri
        print(f"[PHASE 3] Test tahminleri yapılıyor...")

        test_loss, pred_scaled, true_scaled = evaluate(
            model_final, test_loader, torch.nn.MSELoss(), DEVICE
        )

        # Denormalize
        pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        true = target_scaler.inverse_transform(true_scaled.reshape(-1, 1)).flatten()

        # Metrics
        mse, rmse, mae, mape, r2 = regression_metrics(true, pred)

        print(f"Run {run + 1} - RMSE: ${rmse:,.2f}, R²: {r2:.4f}")

        run_results.append({
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
            "predictions": (true, pred),
            "best_epoch": best_epoch,
            "retrain_epochs": retrain_epochs,
        })

    # Ortalama sonuçları hesapla
    avg_rmse = np.mean([r["rmse"] for r in run_results])
    avg_mae = np.mean([r["mae"] for r in run_results])
    avg_mape = np.mean([r["mape"] for r in run_results])
    avg_r2 = np.mean([r["r2"] for r in run_results])

    std_rmse = np.std([r["rmse"] for r in run_results])
    std_r2 = np.std([r["r2"] for r in run_results])

    print(f"\n{'='*70}")
    print(f"{model_name} ORTALAMA SONUÇLAR ({NUM_RUNS} runs):")
    print(f"{'='*70}")
    print(f"  RMSE:  ${avg_rmse:>12,.2f} (±${std_rmse:,.2f})")
    print(f"  MAE:   ${avg_mae:>12,.2f}")
    print(f"  MAPE:  {avg_mape:>11.2f}%")
    print(f"  R²:    {avg_r2:>12.4f} (±{std_r2:.4f})")
    print(f"{'='*70}")

    # En iyi run'ı seç
    best_run = min(run_results, key=lambda x: x["rmse"])

    all_results.append({
        "MODEL": model_name,
        "RMSE": avg_rmse,
        "MAE": avg_mae,
        "MAPE": avg_mape,
        "R2": avg_r2,
        "RMSE_STD": std_rmse,
        "R2_STD": std_r2,
        "best_run": best_run,
        "all_runs": run_results,
    })


# ======================
# Final Comparison
# ======================
print("\n" + "="*70)
print("=== FİNAL KARŞILAŞTIRMA (ORTALAMA) ===")
print("="*70)
print(f"{'Model':<15} {'RMSE ($)':<18} {'MAE ($)':<15} {'MAPE (%)':<12} {'R²':<15}")
print("-" * 70)

for res in all_results:
    print(f"{res['MODEL']:<15} {res['RMSE']:>13,.2f} ±{res['RMSE_STD']:>4,.0f}  "
          f"{res['MAE']:>13,.2f}  {res['MAPE']:>10.2f}  "
          f"{res['R2']:>8.4f} ±{res['R2_STD']:.3f}")

print("="*70)

# Best model
best_idx = min(range(len(all_results)), key=lambda i: all_results[i]["RMSE"])
best_model = all_results[best_idx]["MODEL"]
best_rmse = all_results[best_idx]["RMSE"]
best_r2 = all_results[best_idx]["R2"]

print(f"\n🏆 EN İYİ MODEL: {best_model}")
print(f"   Ortalama RMSE: ${best_rmse:,.2f}")
print(f"   Ortalama R²:   {best_r2:.4f}")
print("="*70)

# Save results
results_df = pd.DataFrame([{
    "MODEL": r["MODEL"],
    "AVG_RMSE": r["RMSE"],
    "STD_RMSE": r["RMSE_STD"],
    "AVG_MAE": r["MAE"],
    "AVG_MAPE": r["MAPE"],
    "AVG_R2": r["R2"],
    "STD_R2": r["R2_STD"],
} for r in all_results])

results_df.to_csv('final_results_ensemble.csv', index=False)
print(f"\n✓ Sonuçlar kaydedildi: final_results_ensemble.csv")

# Save best model's best run predictions
best_result = all_results[best_idx]
best_run = best_result["best_run"]
pred_df = pd.DataFrame({
    'True_Price': best_run['predictions'][0],
    'Predicted_Price': best_run['predictions'][1],
    'Error': best_run['predictions'][1] - best_run['predictions'][0],
    'Percent_Error': ((best_run['predictions'][1] - best_run['predictions'][0]) /
                      best_run['predictions'][0] * 100)
})
pred_df.to_csv(f'{best_model.lower()}_predictions_best.csv', index=False)
print(f"✓ {best_model} tahminleri kaydedildi")

# Detailed run-by-run results
print(f"\n=== {best_model} RUN DETAYLARI ===")
for i, run in enumerate(best_result["all_runs"]):
    print(f"Run {i+1}: RMSE=${run['rmse']:,.2f}, R²={run['r2']:.4f}")

print("\n" + "="*70)
print("=== ÖNCEKİ SONUÇLARLA KARŞILAŞTIRMA ===")
print("="*70)
print("Hocaya gösterdiğiniz sonuçlar:")
print("  MLP:    RMSE = $8,722, R² = 0.78")
print("  LSTM:   RMSE = $7,929, R² = 0.81  ✓ EN İYİ")
print("  Transf: RMSE = $10,190, R² = 0.70")
print("-" * 70)
print(f"Yeni sonuçlar (ensemble ortalaması):")
for res in all_results:
    print(f"  {res['MODEL']:<7} RMSE = ${res['RMSE']:,.0f}, R² = {res['R2']:.2f}")
print("="*70)

if best_rmse < 7929:
    print("\n✅ MÜKEMMEL! LSTM'den daha iyi sonuç!")
elif best_rmse < 8500:
    print("\n✅ ÇOK İYİ! Competitive sonuç!")
elif best_rmse < 9000:
    print("\n⚠️  İYİ ama hala iyileştirilebilir")
else:
    print("\n⚠️  Daha fazla tuning gerekebilir")

print("\n" + "="*70)
print("✅ EĞİTİM TAMAMLANDI!")
print("="*70)