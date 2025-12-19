import torch
import pandas as pd
import numpy as np

from utils.data import load_time_series_dataloaders
from utils.train import train_model, evaluate, regression_metrics, retrain_on_train_val
from models.mlp import ImprovedMLP
from models.lstm import ImprovedLSTM
from models.transformer import TimeSeriesTransformer


# ======================
# Ayarlar
# ======================
CSV_PATH = "dc_extended.csv"
WINDOW_SIZE = 90
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print("="*70)
print("BITCOIN FİYAT TAHMİNİ PROJESİ")
print("="*70)


# ======================
# VERİ YÜKLEME
# ======================
print("\n1. VERİ YÜKLEME")
print("-" * 70)

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

print(f"✓ Pencere boyutu: {seq_len} gün")
print(f"✓ Feature sayısı: {num_features}")
print(f"✓ Train: {len(train_loader.dataset)} örnek")
print(f"✓ Val:   {len(val_loader.dataset)} örnek")
print(f"✓ Test:  {len(test_loader.dataset)} örnek")


# ======================
# MODEL TANIMLARI
# ======================
print("\n2. MODEL TANIMLARI")
print("-" * 70)

model_configs = {
    "MLP": {
        "class": ImprovedMLP,
        "kwargs": {"seq_len": seq_len, "num_features": num_features, "dropout": 0.3},
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "description": "4-layer MLP with BatchNorm"
    },

    "LSTM": {
        "class": ImprovedLSTM,
        "kwargs": {"num_features": num_features, "hidden_size": 128,
                   "num_layers": 3, "dropout": 0.3},
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "description": "Bidirectional LSTM (3 layers) + Attention"
    },

    "Transformer": {
        "class": TimeSeriesTransformer,
        "kwargs": {"num_features": num_features, "d_model": 64, "nhead": 4,
                   "num_layers": 2, "dim_feedforward": 256, "dropout": 0.3},
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "description": "CNN + Transformer Encoder (2 layers)"
    },
}

for name, config in model_configs.items():
    print(f"✓ {name}: {config['description']}")


# ======================
# MODEL EĞİTİMİ
# ======================
print("\n3. MODEL EĞİTİMİ")
print("="*70)

all_results = []

for model_name, config in model_configs.items():
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")

    # PHASE 1: Train ve Validation ile eğitim
    print(f"\n[PHASE 1] Train ve Val ile eğitim başlıyor...")

    model = config["class"](**config["kwargs"])

    model_trained, train_losses, val_losses, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        patience=PATIENCE,
        name=f"{model_name.lower()}_phase1",
        device=DEVICE,
        scheduler_type="plateau",
        return_best_epoch=True,
    )

    # PHASE 2: Train + Val birleştir ve retrain
    retrain_epochs = int(best_epoch * 1.5)
    retrain_epochs = max(retrain_epochs, 30)
    retrain_epochs = min(retrain_epochs, 100)

    print(f"\n[PHASE 2] Train+Val birleştiriliyor ve retrain yapılıyor...")

    model_final = config["class"](**config["kwargs"])

    model_final = retrain_on_train_val(
        model=model_final,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=retrain_epochs,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        name=f"{model_name.lower()}_final",
        device=DEVICE,
    )

    # PHASE 3: Test seti ile tahmin
    print(f"\n[PHASE 3] Test tahminleri yapılıyor...")

    test_loss, pred_scaled, true_scaled = evaluate(
        model_final, test_loader, torch.nn.MSELoss(), DEVICE
    )

    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    true = target_scaler.inverse_transform(true_scaled.reshape(-1, 1)).flatten()

    mse, rmse, mae, mape, r2 = regression_metrics(true, pred)

    print(f"\n{model_name} SONUÇLARI")
    print(f"  RMSE:        ${rmse:>12,.2f}")
    print(f"  MAE:         ${mae:>12,.2f}")
    print(f"  MAPE:        {mape:>11.2f}%")
    print(f"  R²:          {r2:>12.4f}")

    all_results.append({
        "MODEL": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "best_epoch": best_epoch,
        "retrain_epochs": retrain_epochs,
        "predictions": (true, pred),
    })


# ======================
# KARŞILAŞTIRMA
# ======================
print("\n" + "="*70)
print("4. TÜM MODELLERİN KARŞILAŞTIRILMASI")
print("="*70)
print(f"\n{'Model':<15} {'RMSE ($)':<15} {'MAE ($)':<15} {'MAPE (%)':<12} {'R²':<10}")
print("-" * 70)

for res in all_results:
    print(f"{res['MODEL']:<15} {res['RMSE']:>13,.2f}  "
          f"{res['MAE']:>13,.2f}  {res['MAPE']:>10.2f}  {res['R2']:>8.4f}")

results_df = pd.DataFrame([{
    "MODEL": r["MODEL"],
    "RMSE": r["RMSE"],
    "MAE": r["MAE"],
    "MAPE": r["MAPE"],
    "R2": r["R2"]
} for r in all_results])

best_idx = results_df['RMSE'].idxmin()
best_model = results_df.loc[best_idx, 'MODEL']
best_rmse = results_df.loc[best_idx, 'RMSE']
best_r2 = results_df.loc[best_idx, 'R2']

print(f"\n🏆 EN İYİ MODEL: {best_model}")
print(f"   RMSE: ${best_rmse:,.2f}")
print(f"   R²:   {best_r2:.4f}")


# ======================
# SONUÇLARI KAYDET
# ======================
print("\n5. SONUÇLARI KAYDET")
results_df.to_csv('model_comparison.csv', index=False)

best_result = all_results[best_idx]
pred_df = pd.DataFrame({
    'True_Price': best_result['predictions'][0],
    'Predicted_Price': best_result['predictions'][1],
    'Error': best_result['predictions'][1] - best_result['predictions'][0]
})
pred_df.to_csv(f'{best_model.lower()}_predictions.csv', index=False)

print("\nPROJE TAMAMLANDI!")