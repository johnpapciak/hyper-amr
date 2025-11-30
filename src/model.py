"""Hyperbolic AMR model components lifted from the exploratory notebook."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# --------------------------- Geometry helpers ---------------------------

def project_to_ball(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    norm = torch.norm(x, dim=-1, keepdim=True)
    max_norm = 1.0 - eps
    scale = torch.clamp(max_norm / norm, max=1.0)
    return x * scale


def exp_map_zero(v: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
   norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=eps)
   c_t = v.new_tensor(c)               # tensor with same device/dtype as v
   sqrt_c = torch.sqrt(c_t)
   factor = torch.tanh(sqrt_c * norm) / (sqrt_c * norm)
   return project_to_ball(v * factor)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c * c * x2 * y2
    return project_to_ball(num / denom.clamp(min=eps))


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
   x2 = torch.sum(x * x, dim=-1, keepdim=True)
   y2 = torch.sum(y * y, dim=-1, keepdim=True)
   xy = torch.sum((x - y) * (x - y), dim=-1, keepdim=True)
   c_t = xy.new_tensor(c)              # tensor on same device/dtype
   sqrt_c = torch.sqrt(c_t)
   num = 2 * sqrt_c * torch.sqrt(xy)
   denom = (1 - c_t * x2).clamp(min=eps) * (1 - c_t * y2).clamp(min=eps)

   finite_mask = (
       torch.isfinite(x2)
       & torch.isfinite(y2)
       & torch.isfinite(xy)
       & torch.isfinite(num)
       & torch.isfinite(denom)
   )

   acosh_arg = torch.clamp_min(1 + num / denom, 1 + eps)
   dist = torch.acosh(acosh_arg).squeeze(-1)
   return torch.where(finite_mask.squeeze(-1), dist, torch.zeros_like(dist))


def info_nce_hyper(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Contrastive alignment using hyperbolic distance as similarity."""

    B = z1.size(0)
    d = poincare_distance(z1.unsqueeze(1), z2.unsqueeze(0))  # (B,B)
    logits = -d / temperature
    labels = torch.arange(B, device=z1.device)
    return nn.CrossEntropyLoss()(logits, labels)


def stacked_entailment_radial(z_parent: torch.Tensor, z_child: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    """Encourage child embeddings to sit farther from the origin than parents."""

    r_par = torch.norm(z_parent, dim=-1)
    r_child = torch.norm(z_child, dim=-1)
    return torch.relu(margin + r_par - r_child).mean()


def bce_with_pos_weight(logits: torch.Tensor, targets: torch.Tensor, pos_weight: Optional[torch.Tensor]) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


# ------------------------------- Dataset --------------------------------

class ContigDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, T: Optional[np.ndarray] = None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)
        if T is None:
            T = np.zeros(len(X), dtype=np.int64)
        self.T = torch.as_tensor(T, dtype=torch.long)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.X.shape[0]

    def __getitem__(self, i: int):  # pragma: no cover - trivial
        return self.X[i], self.Y[i], self.T[i]


# ------------------------------- Model ----------------------------------

@dataclass
class ModelConfig:
    seq_dim: int
    amr_classes: int
    taxonomy_size: int = 0
    hidden: int = 512
    embed_dim: int = 128
    use_taxonomy: bool = False
    taxonomy_levels: int = 1


class HyperAMR(nn.Module):
    """Lightweight hyperbolic AMR model suitable for CLI training."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.seq_encoder = nn.Sequential(
            nn.Linear(cfg.seq_dim, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.embed_dim),
        )
        self.amr_encoder = nn.Sequential(
            nn.Linear(cfg.amr_classes, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.embed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.amr_classes),
        )

        self.use_taxonomy = cfg.use_taxonomy and cfg.taxonomy_size > 0
        self.taxonomy_levels = cfg.taxonomy_levels
        if self.use_taxonomy:
            self.taxon_embed = nn.Embedding(cfg.taxonomy_size, cfg.embed_dim)
        else:
            self.taxon_embed = None

    def forward(
        self,
        seq_feats: torch.Tensor,
        amr_targets: torch.Tensor,
        tax_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z_seq_euc = self.seq_encoder(seq_feats)
        z_amr_euc = self.amr_encoder(amr_targets)

        z_seq = project_to_ball(exp_map_zero(z_seq_euc))
        z_amr = project_to_ball(exp_map_zero(z_amr_euc))

        amr_logits = self.decoder(z_seq)
        out = {"z_seq": z_seq, "z_amr": z_amr, "amr_logits": amr_logits}

        if self.use_taxonomy and tax_idx is not None and self.taxon_embed is not None:
            if tax_idx.dim() == 1:
                tax_vec = self.taxon_embed(tax_idx.clamp_min(0))
                out["z_tax"] = project_to_ball(exp_map_zero(tax_vec))
                out["z_tax_leaf"] = out["z_tax"]
            elif tax_idx.dim() == 2:
                B, L = tax_idx.shape
                tax_idx_clean = tax_idx.clone()
                mask_missing = tax_idx_clean < 0
                tax_idx_clean = tax_idx_clean.clamp_min(0)

                z_tax_euc = self.taxon_embed(tax_idx_clean.view(-1))
                z_tax = project_to_ball(exp_map_zero(z_tax_euc)).view(B, L, self.cfg.embed_dim)

                if mask_missing.any():
                    z_tax = z_tax.masked_fill(mask_missing.unsqueeze(-1), 0.0)

                out["z_tax_lineage"] = z_tax
                out["z_tax_leaf"] = z_tax[:, -1, :]
            else:
                raise ValueError(f"tax_idx must have dim 1 or 2, got {tax_idx.dim()}")
        return out


# ----------------------------- Train / Eval -----------------------------

def build_dataloaders(
    X: np.ndarray,
    Y: np.ndarray,
    split: np.ndarray,
    taxonomy: Optional[np.ndarray] = None,
    taxonomy_size: int | None = None,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    _validate_taxonomy_indices(taxonomy, taxonomy_size)

    idx_train = np.where(split == "train")[0]
    idx_val = np.where(split == "val")[0]
    idx_test = np.where(split == "test")[0]

    def subset(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return arr[idx] if arr is not None else None

    train_ds = ContigDataset(X[idx_train], Y[idx_train], subset(taxonomy, idx_train))
    val_ds = ContigDataset(X[idx_val], Y[idx_val], subset(taxonomy, idx_val))
    test_ds = ContigDataset(X[idx_test], Y[idx_test], subset(taxonomy, idx_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def _validate_taxonomy_indices(taxonomy: Optional[np.ndarray], taxonomy_size: int | None):
    """Fail fast when taxonomy indices fall outside the configured range."""

    if taxonomy is None or taxonomy_size is None:
        return

    if taxonomy_size <= 0:
        raise ValueError("taxonomy_size must be positive when taxonomy indices are provided")

    tax = np.asarray(taxonomy)
    if tax.size == 0:
        return

    invalid_nan = np.isnan(tax)
    invalid_negative = tax < 0
    invalid_large = tax >= taxonomy_size

    invalid_mask = invalid_nan | invalid_negative | invalid_large
    if invalid_mask.any():
        coords = np.argwhere(invalid_mask)
        summary = []
        for label, mask in ("NaN", invalid_nan), ("<0", invalid_negative), (">= taxonomy_size", invalid_large):
            count = int(mask.sum())
            if count:
                summary.append(f"{label}: {count}")

        samples = coords[:10].tolist()
        msg = (
            "Invalid taxonomy indices detected before DataLoader construction. "
            + "; ".join(summary)
            + f"; first offending indices (row, col): {samples}"
        )
        raise ValueError(msg)


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_align: float = 1.0
    lambda_bce: float = 1.0
    lambda_tax: float = 0.0
    progress_bar: bool = True
    log_batch_progress: bool = True
    log_every: int = 50


@dataclass
class TrainState:
    history: list[dict]
    class_weights: Optional[torch.Tensor]


def compute_class_weights(Y_train: np.ndarray) -> torch.Tensor:
    pos = Y_train.sum(axis=0) + 1e-3
    neg = Y_train.shape[0] - pos + 1e-3
    return torch.from_numpy((neg / pos).astype(np.float32))


def train_model(
    model: HyperAMR,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    class_weights: Optional[torch.Tensor] = None,
) -> TrainState:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: list[dict] = []

    for ep in range(1, cfg.epochs + 1):
        tr = _run_epoch(model, train_loader, opt, cfg, class_weights, train=True, device=device)
        va = _run_epoch(model, val_loader, None, cfg, class_weights, train=False, device=device)
        rec = {"epoch": ep, "train": tr, "val": va}
        history.append(rec)
        print(
            f"[{ep:02d}] train loss={tr['loss']:.4f} bce={tr['bce']:.4f} align={tr['align']:.4f} | "
            f"val loss={va['loss']:.4f} bce={va['bce']:.4f} align={va['align']:.4f}"
        )
    return TrainState(history=history, class_weights=class_weights)


def _run_epoch(
    model: HyperAMR,
    loader: DataLoader,
    opt: Optional[torch.optim.Optimizer],
    cfg: TrainConfig,
    class_weights: Optional[torch.Tensor],
    train: bool,
    device: torch.device,
) -> dict:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = total_bce = total_align = total_tax = 0.0
    n_obs = 0
    cw = class_weights.to(device) if class_weights is not None else None

    progress_bar = None
    iterator = loader
    mode = "train" if train else "eval"
    if cfg.progress_bar:
        from tqdm import tqdm

        progress_bar = tqdm(loader, desc=f"{mode} batches", leave=False)
        iterator = progress_bar

    for batch_idx, (xb, yb, tb) in enumerate(iterator, start=1):
        xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
        if train:
            opt.zero_grad()

        out = model(xb, yb, tax_idx=tb)
        loss_align = info_nce_hyper(out["z_seq"], out["z_amr"])
        loss_bce = bce_with_pos_weight(out["amr_logits"], yb, pos_weight=cw)
        loss_tax = torch.tensor(0.0, device=device)
        if model.use_taxonomy and cfg.lambda_tax > 0.0:
            if "z_tax_lineage" in out:
                z_line = out["z_tax_lineage"]  # [B, L, D]
                B, L, D = z_line.shape
                if L > 1:
                    parents = z_line[:, :-1, :].reshape(-1, D)
                    childs = z_line[:, 1:, :].reshape(-1, D)
                    p_norm = parents.norm(dim=-1)
                    c_norm = childs.norm(dim=-1)
                    valid_mask = (p_norm > 0) & (c_norm > 0)
                    if valid_mask.any():
                        loss_tax = stacked_entailment_radial(
                            parents[valid_mask], childs[valid_mask], margin=0.05
                        )
            elif "z_tax" in out:
                loss_tax = stacked_entailment_radial(out["z_seq"], out["z_tax"], margin=0.05)

        loss = cfg.lambda_align * loss_align + cfg.lambda_bce * loss_bce + cfg.lambda_tax * loss_tax

        if not torch.isfinite(loss) or not torch.isfinite(loss_align) or not torch.isfinite(loss_bce) or not torch.isfinite(loss_tax):
            logging.warning("Skipping batch %d due to non-finite loss components", batch_idx)
            continue

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        bs = xb.size(0)
        n_obs += bs
        total_loss += loss.item() * bs
        total_bce += loss_bce.item() * bs
        total_align += loss_align.item() * bs
        total_tax += loss_tax.item() * bs

        if progress_bar is not None:
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                bce=f"{loss_bce.item():.4f}",
                align=f"{loss_align.item():.4f}",
                tax=f"{loss_tax.item():.4f}",
            )

        if cfg.log_batch_progress and cfg.log_every > 0 and batch_idx % cfg.log_every == 0:
            logging.info(
                "%s batch %d/%d loss=%.4f bce=%.4f align=%.4f tax=%.4f",
                mode,
                batch_idx,
                len(loader),
                loss.item(),
                loss_bce.item(),
                loss_align.item(),
                loss_tax.item(),
            )

    if progress_bar is not None:
        progress_bar.close()

    if n_obs == 0:
        logging.warning("%s epoch produced no valid observations; returning NaN metrics", mode)
        return {"loss": float("nan"), "bce": float("nan"), "align": float("nan"), "tax": float("nan")}

    return {
        "loss": total_loss / n_obs,
        "bce": total_bce / n_obs,
        "align": total_align / n_obs,
        "tax": total_tax / n_obs,
    }


@dataclass
class EvalResults:
    logits: np.ndarray
    targets: np.ndarray
    predictions: np.ndarray
    thresholds: np.ndarray
    macro_aupr: float
    macro_auroc: float
    aupr_per_class: list[float]
    auroc_per_class: list[float]
    accuracy_per_class: list[float]
    overall_accuracy: float


def _collect_logits_and_targets(
    model: HyperAMR, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    logits_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []

    with torch.no_grad():
        for xb, yb, tb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb, yb)
            logits_all.append(out["amr_logits"].cpu().numpy())
            targets_all.append(yb.cpu().numpy())

    logits = np.concatenate(logits_all, axis=0)
    targets = np.concatenate(targets_all, axis=0)
    return logits, targets


def _search_thresholds(
    model: HyperAMR,
    loader: DataLoader,
    device: torch.device,
    metric: str = "f1",
    default: float = 0.5,
) -> np.ndarray:
    from sklearn.metrics import average_precision_score, f1_score

    logits, targets = _collect_logits_and_targets(model, loader, device)
    probs = 1 / (1 + np.exp(-logits))
    C = targets.shape[1]
    thresholds = np.full(C, default, dtype=np.float32)
    candidate_thresholds = np.linspace(0.0, 1.0, 101)

    for c in range(C):
        y_true = targets[:, c]
        y_prob = probs[:, c]
        if y_true.sum() == 0 or (y_true == 0).sum() == 0:
            # Degenerate class; keep default threshold.
            continue

        best_score = float("-inf")
        best_thr = default
        for thr in candidate_thresholds:
            preds = (y_prob >= thr).astype(np.float32)
            if metric == "f1":
                score = f1_score(y_true, preds, zero_division=0)
            elif metric == "average_precision":
                score = average_precision_score(y_true, preds)
            else:
                raise ValueError(f"Unsupported calibration metric: {metric}")

            if score > best_score:
                best_score = score
                best_thr = float(thr)

        thresholds[c] = best_thr

    return thresholds


def evaluate(
    model: HyperAMR,
    loader: DataLoader,
    class_list: Iterable[str],
    thresholds: Optional[np.ndarray] = None,
    calibrate_thresholds: bool = False,
    calibration_loader: Optional[DataLoader] = None,
    calibration_metric: str = "f1",
) -> EvalResults:
    from sklearn.metrics import average_precision_score, roc_auc_score

    class_list = list(class_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    if thresholds is None:
        if calibrate_thresholds:
            if calibration_loader is None:
                raise ValueError("calibration_loader must be provided when calibrating thresholds")
            thresholds = _search_thresholds(
                model, calibration_loader, device, metric=calibration_metric
            )
        else:
            thresholds = np.full(len(class_list), 0.5, dtype=np.float32)

    logits, targets = _collect_logits_and_targets(model, loader, device)
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= thresholds).astype(np.float32)

    aupr_per_class = []
    auroc_per_class = []
    accuracy_per_class = []
    C = targets.shape[1]
    for c in range(C):
        y_true = targets[:, c]
        y_prob = probs[:, c]
        y_pred = preds[:, c]
        accuracy_per_class.append(float((y_pred == y_true).mean()))
        if y_true.sum() == 0 or (y_true == 0).sum() == 0:
            aupr_per_class.append(np.nan)
            auroc_per_class.append(np.nan)
            continue
        aupr_per_class.append(average_precision_score(y_true, y_prob))
        try:
            auroc_per_class.append(roc_auc_score(y_true, y_prob))
        except ValueError:
            auroc_per_class.append(np.nan)

    return EvalResults(
        logits=logits,
        targets=targets,
        predictions=preds,
        thresholds=thresholds,
        macro_aupr=float(np.nanmean(aupr_per_class)),
        macro_auroc=float(np.nanmean(auroc_per_class)),
        aupr_per_class=aupr_per_class,
        auroc_per_class=auroc_per_class,
        accuracy_per_class=accuracy_per_class,
        overall_accuracy=float((preds == targets).mean()),
    )


