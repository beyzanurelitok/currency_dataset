# utils/plots.py
"""
Visualization utilities for project report
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_curves(train_losses, val_losses, model_name, save_path="plots"):
    """
    Plot and save training/validation loss curves
    """
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(f'{model_name} - Training vs Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Mark best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = val_losses[best_epoch - 1]
    plt.plot(best_epoch, best_val_loss, 'g*', markersize=15,
             label=f'Best (Epoch {best_epoch})')
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{save_path}/{model_name}_training_curves.png", dpi=300)
    plt.close()

    print(f"✓ Saved: {save_path}/{model_name}_training_curves.png")


def plot_predictions(y_true, y_pred, model_name, save_path="plots"):
    """
    Plot actual vs predicted values
    """
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(14, 6))

    # Line plot
    plt.subplot(1, 2, 1)
    indices = range(len(y_true))
    plt.plot(indices, y_true, 'b-', label='Actual', alpha=0.7, linewidth=1.5)
    plt.plot(indices, y_pred, 'r--', label='Predicted', alpha=0.7, linewidth=1.5)
    plt.xlabel('Sample Index', fontsize=11)
    plt.ylabel('Price (USD)', fontsize=11)
    plt.title(f'{model_name} - Actual vs Predicted', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--',
             label='Perfect Prediction', linewidth=2)

    plt.xlabel('Actual Price (USD)', fontsize=11)
    plt.ylabel('Predicted Price (USD)', fontsize=11)
    plt.title(f'{model_name} - Scatter Plot', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}/{model_name}_predictions.png", dpi=300)
    plt.close()

    print(f"✓ Saved: {save_path}/{model_name}_predictions.png")


def plot_error_distribution(y_true, y_pred, model_name, save_path="plots"):
    """
    Plot error distribution histogram
    """
    os.makedirs(save_path, exist_ok=True)

    errors = y_pred - y_true
    percentage_errors = (errors / y_true) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute errors
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel('Prediction Error (USD)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title(f'{model_name} - Error Distribution', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Percentage errors
    axes[1].hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].set_xlabel('Percentage Error (%)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title(f'{model_name} - Percentage Error Distribution', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}/{model_name}_error_distribution.png", dpi=300)
    plt.close()

    print(f"✓ Saved: {save_path}/{model_name}_error_distribution.png")


def plot_all_models_comparison(results_df, save_path="plots"):
    """
    Compare all models in a single bar chart
    """
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = results_df['MODEL']
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    titles = ['RMSE (Lower is Better)', 'MAE (Lower is Better)',
              'MAPE (Lower is Better)', 'R² Score (Higher is Better)']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]

        values = results_df[metric]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        bars = ax.bar(models, values, color=colors[:len(models)], alpha=0.8, edgecolor='black')

        # Highlight best model
        if metric == 'R2':
            best_idx = values.idxmax()
        else:
            best_idx = values.idxmin()
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('darkgoldenrod')
        bars[best_idx].set_linewidth(3)

        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.grid(True, axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}' if metric != 'RMSE' else f'${height:,.0f}',
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{save_path}/all_models_comparison.png", dpi=300)
    plt.close()

    print(f"✓ Saved: {save_path}/all_models_comparison.png")


def create_all_plots(results_dict, save_path="plots"):
    """
    Create all plots for the report

    Args:
        results_dict: Dictionary with keys:
            - 'models': list of model names
            - 'train_losses': list of training loss arrays
            - 'val_losses': list of validation loss arrays
            - 'predictions': list of (y_true, y_pred) tuples
            - 'results_df': DataFrame with final metrics
    """
    os.makedirs(save_path, exist_ok=True)
    print("\n" + "="*60)
    print("CREATING PLOTS FOR REPORT")
    print("="*60)

    # 1. Training curves for each model
    for name, train_loss, val_loss in zip(
        results_dict['models'],
        results_dict['train_losses'],
        results_dict['val_losses']
    ):
        plot_training_curves(train_loss, val_loss, name, save_path)

    # 2. Predictions for each model
    for name, (y_true, y_pred) in zip(
        results_dict['models'],
        results_dict['predictions']
    ):
        plot_predictions(y_true, y_pred, name, save_path)
        plot_error_distribution(y_true, y_pred, name, save_path)

    # 3. Overall comparison
    plot_all_models_comparison(results_dict['results_df'], save_path)

    print("\n✓ All plots created successfully!")
    print(f"✓ Plots saved in '{save_path}/' directory")