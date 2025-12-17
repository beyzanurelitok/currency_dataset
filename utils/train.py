# utils/train.py

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()  # Remove extra dimensions
        loss = criterion(y_pred, y_batch)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
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

            y_pred = model(X_batch).squeeze()  # Remove extra dimensions
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
    scheduler_type: str = "plateau"  # Options: "plateau", "cosine", "step", "none"
):
    """
    Train a model with flexible scheduler options.

    Args:
        scheduler_type:
            - "plateau": ReduceLROnPlateau (RECOMMENDED - adaptive, safe)
            - "cosine": CosineAnnealingWarmRestarts (aggressive, for large datasets)
            - "step": StepLR (simple, predictable)
            - "none": No scheduler (constant LR)
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Choose scheduler
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,      # Reduce LR by half
            patience=10,     # Wait 10 epochs before reducing
            min_lr=1e-6
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5
        )
    else:  # "none"
        scheduler = None

    best_val_loss = float("inf")
    best_state = None
    train_losses, val_losses = [], []
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(val_loss)  # Plateau needs metric
            else:
                scheduler.step()  # Others step automatically

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print every 10 epochs to reduce clutter
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch}: train={train_loss:.6f}, val={val_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping logic
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model
    torch.save(model.state_dict(), f"{name}.pt")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return model, train_losses, val_losses


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
    Warmup helps stabilize training at the beginning.
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

    # Main scheduler (after warmup)
    main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6
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

        # Print progress
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


def regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    Note: y_true and y_pred should be in ORIGINAL scale (after inverse_transform)
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return mse, rmse, mae, mape, r2


# Ablation study helper
def compare_schedulers(model_class, model_kwargs, train_loader, val_loader, test_loader,
                       target_scaler, device, epochs=100):
    """
    Compare different scheduler strategies on the same model architecture.
    Useful for ablation studies in your report.
    """
    schedulers = ["none", "step", "plateau", "cosine"]
    results = []

    for sched_type in schedulers:
        print(f"\n{'='*60}")
        print(f"Testing scheduler: {sched_type.upper()}")
        print(f"{'='*60}")

        # Create fresh model
        model = model_class(**model_kwargs)

        # Train
        model_trained, train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=1e-3,
            weight_decay=1e-5,
            patience=20,
            name=f"scheduler_test_{sched_type}",
            device=device,
            scheduler_type=sched_type
        )

        # Evaluate
        _, pred_scaled, true_scaled = evaluate(
            model_trained, test_loader, torch.nn.MSELoss(), device
        )

        pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        true = target_scaler.inverse_transform(true_scaled.reshape(-1, 1)).flatten()

        _, rmse, mae, mape, r2 = regression_metrics(true, pred)

        results.append({
            "scheduler": sched_type,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
            "train_losses": train_losses,
            "val_losses": val_losses
        })

        print(f"✓ RMSE: ${rmse:,.2f}, R²: {r2:.4f}\n")

    return results