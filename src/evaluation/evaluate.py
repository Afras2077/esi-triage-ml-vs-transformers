"""
Shared evaluation utilities — metrics, confusion matrix, comparison plots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, cohen_kappa_score
)
from pathlib import Path


def compute_metrics(y_true, y_pred, model_name: str = "") -> dict:
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred, weights="linear"),
    }


def plot_confusion_matrix(y_true, y_pred, labels, model_name: str, out_dir: str = "results/figures/"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("True ESI"); ax.set_xlabel("Predicted ESI")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_dir}cm_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare_models(results: list[dict], out_dir: str = "results/figures/"):
    df = pd.DataFrame(results).set_index("model")
    df[["accuracy", "weighted_f1", "macro_f1", "cohen_kappa"]].plot(kind="bar", figsize=(12, 5))
    plt.title("Model Comparison — ESI Triage Classification")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{out_dir}model_comparison.png", dpi=150)
    plt.close()
