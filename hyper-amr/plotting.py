"""Plotting utilities extracted from the exploratory notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, auc


def plot_class_frequency(Y_amr: np.ndarray, class_list: Iterable[str], output_path: Path | None = None):
    freq = np.sum(Y_amr, axis=0)
    freq_df = pd.DataFrame({"AMR_Class": list(class_list), "Count": freq}).sort_values("Count", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(freq_df["AMR_Class"], freq_df["Count"])
    plt.xticks(rotation=90)
    plt.xlabel("AMR class")
    plt.ylabel("Contigs")
    plt.title("AMR Class Frequency")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
    plt.close()
    return freq_df


def plot_pr_curves(logits: np.ndarray, targets: np.ndarray, class_list: list[str], output_path: Path | None = None):
    probs = 1.0 / (1.0 + np.exp(-logits))
    supports = targets.sum(axis=0).astype(int)
    valid_mask = supports > 0
    valid_idx = [i for i, v in enumerate(valid_mask) if v]
    if not valid_idx:
        raise ValueError("No classes with positives in the provided targets")

    top_idx = sorted(valid_idx, key=lambda i: supports[i], reverse=True)[: min(5, len(valid_idx))]

    y_true_micro = targets[:, valid_mask].ravel()
    y_prob_micro = probs[:, valid_mask].ravel()
    p_micro, r_micro, _ = precision_recall_curve(y_true_micro, y_prob_micro)
    ap_micro = average_precision_score(y_true_micro, y_prob_micro)

    plt.figure(figsize=(8, 6))
    plt.plot(r_micro, p_micro, linewidth=2, label=f"micro-average (AP={ap_micro:.3f})")

    for i in top_idx:
        y_true = targets[:, i]
        y_prob = probs[:, i]
        p, r, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(r, p, linewidth=1.5, label=f"{class_list[i]} (AP={ap:.3f}, n+={supports[i]})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves (Top Test Classes)")
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
    plt.close()

    summary = []
    for i in top_idx:
        y_true = targets[:, i]
        y_prob = probs[:, i]
        ap = average_precision_score(y_true, y_prob)
        summary.append((class_list[i], int(supports[i]), ap))
    return {
        "micro_ap": ap_micro,
        "top_classes": summary,
    }


