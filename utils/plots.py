# utils/plots.py

import matplotlib.pyplot as plt
import os


def plot_loss_curves(
    mlp_train, mlp_val,
    lstm_train, lstm_val,
    trf_train, trf_val,
    out_dir: str = "."
):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(mlp_train, label="MLP train")
    plt.plot(mlp_val, label="MLP val")
    plt.plot(lstm_train, label="LSTM train")
    plt.plot(lstm_val, label="LSTM val")
    plt.plot(trf_train, label="Transformer train")
    plt.plot(trf_val, label="Transformer val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves_all.png"), dpi=200)
    plt.close()


def plot_true_vs_pred(y_true, y_pred, title: str, out_path: str):
    plt.figure()
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.xlabel("Test Sample Index")
    plt.ylabel("close_USD")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scatter(y_true, y_pred, title: str, out_path: str):
    plt.figure()
    plt.scatter(y_true, y_pred, s=10, alpha=0.7)
    plt.xlabel("True close_USD")
    plt.ylabel("Predicted close_USD")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
