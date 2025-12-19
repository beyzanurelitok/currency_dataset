# utils/train.py

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import ConcatDataset, DataLoader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)

            preds.append(y_pred.cpu().numpy())
            targets.append(y_batch.cpu().numpy())

    preds = np.concatenate(preds).flatten()
    targets = np.concatenate(targets).flatten()

    return total_loss / len(loader.dataset), preds, targets


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    name: str,
    device,
    scheduler_type: str = "plateau",
    return_best_epoch: bool = False,  # YENİ: Best epoch döndürsün mü?
):
    """
    Train a model with early stopping.

    Args:
        return_best_epoch: If True, returns (model, train_losses, val_losses, best_epoch)
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        scheduler = None

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0  # YENİ: En iyi epoch'u kaydet
    train_losses, val_losses = [], []
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}: train={train_loss:.6f}, val={val_loss:.6f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = model.state_dict()
            best_epoch = epoch  # YENİ: Güncelle
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⚠ Early stopping at epoch {epoch}")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model
    torch.save(model.state_dict(), f"{name}.pt")
    print(f"  ✓ Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

    if return_best_epoch:
        return model, train_losses, val_losses, best_epoch
    else:
        return model, train_losses, val_losses


def retrain_on_train_val(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    name: str,
    device,
):
    """
    YENİ FONKSIYON: Train ve Val'i birleştirip modeli yeniden eğit.

    Hocalarınızın önerisi: "train ve validation ile işiniz bittiğinde
    zamansal olarak birleştirip testdeki tahminler o şekilde yapılmalı"

    Args:
        model: Eğitilecek model (fresh initialization)
        train_loader: Train dataloader
        val_loader: Validation dataloader
        epochs: Kaç epoch eğitilecek (genelde best_epoch kullanılır)
        lr, weight_decay: Hiperparametreler
        name: Model ismi
        device: CPU/CUDA

    Returns:
        Eğitilmiş model
    """
    print(f"  → Train ve Val birleştiriliyor...")

    # Train ve Val dataset'lerini birleştir
    combined_dataset = ConcatDataset([
        train_loader.dataset,
        val_loader.dataset
    ])

    # Yeni dataloader (aynı batch size)
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,  # ÖNEMLİ: Shuffle yapıyoruz çünkü zaten zamansal sırada birleştirdik
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )

    print(f"  ✓ Toplam {len(combined_dataset)} örnek ile eğitim başlıyor")
    print(f"  ✓ {epochs} epoch boyunca eğitilecek")

    # Model eğitimi
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler (opsiyonel, genelde best_epoch zaten bulunduğu için gerek yok)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=max(epochs // 5, 5), min_lr=1e-6
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in combined_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        epoch_loss = total_loss / len(combined_dataset)
        scheduler.step(epoch_loss)

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Retrain Epoch {epoch:>3}/{epochs}: loss={epoch_loss:.6f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

    # Save final model
    torch.save(model.state_dict(), f"{name}.pt")
    print(f"  ✓ Final model saved: {name}.pt")

    return model


def regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100

    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return mse, rmse, mae, mape, r2


def train_model_with_warmup(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    name: str,
    device,
    warmup_epochs: int = 5
):
    """
    Advanced training with learning rate warmup + ReduceLROnPlateau.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Warmup scheduler
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_state = None
    train_losses, val_losses = [], []
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        # Step schedulers
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            status = "WARMUP" if epoch <= warmup_epochs else "TRAINING"
            print(f"{epoch} [{status}]: train={train_loss:.6f}, val={val_loss:.6f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), f"{name}.pt")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return model, train_losses, val_losses