"""Command-line interface that mirrors the notebook workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data as dutils
from . import model as mutils
from . import plotting


def _json_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def cmd_download(args: argparse.Namespace):
    urls = _json_list(args.urls)
    paths = dutils.download_genomes(urls, Path(args.output))
    for p in paths:
        print(p)


def cmd_run_amrfinder(args: argparse.Namespace):
    out = Path(args.output)
    db = Path(args.database) if args.database else None

    if args.run_many:
        fastas = [Path(p) for p in _json_list(args.fasta)]
        if not fastas:
            raise ValueError("No FASTA paths provided; supply a comma-delimited list with --fasta")
        for fasta in fastas:
            tsv = dutils.run_amrfinder_on_fasta(fasta, out, threads=args.threads, db=db)
            print(tsv)
    else:
        fasta = Path(args.fasta)
        tsv = dutils.run_amrfinder_on_fasta(fasta, out, threads=args.threads, db=db)
        print(tsv)


def cmd_prepare(args: argparse.Namespace):
    tsvs = [Path(p) for p in _json_list(args.tsvs)]
    fastas = [Path(p) for p in _json_list(args.fastas)]
    out_dir = Path(args.output)

    artifacts = dutils.build_amr_labels(tsvs, fastas, out_dir)
    df = pd.read_parquet(artifacts.labels_path)
    df = dutils.attach_sequences(df, fastas)
    df = dutils.train_val_test_split(df)
    df.to_parquet(artifacts.labels_path, index=False)
    print(f"Saved labels -> {artifacts.labels_path}")
    print(f"Saved classes -> {artifacts.classes_path}")


def cmd_train(args: argparse.Namespace):
    artifacts_dir = Path(args.artifacts)
    labels_path = artifacts_dir / "contig_amr_labels.parquet"
    classes_path = artifacts_dir / "amr_class_list.json"

    df = pd.read_parquet(labels_path)
    with open(classes_path) as f:
        class_list = json.load(f)["class_list"]

    cfg = dutils.HashingConfig(k=args.k, buckets=args.buckets, stride=args.stride, max_len=args.max_len)
    df = dutils.attach_sequences(df, [Path(p) for p in _json_list(args.fastas)])
    X_seq, Y_amr = dutils.prepare_features(df, class_list, cfg, threads=args.hash_threads)

    split = df["split"].values if "split" in df.columns else dutils.train_val_test_split(df)["split"].values
    taxonomy = df["taxid"].astype(str).values if "taxid" in df.columns else None

    train_loader, val_loader, test_loader = mutils.build_dataloaders(
        X_seq, Y_amr, split, taxonomy, num_workers=args.loader_workers
    )

    mcfg = mutils.ModelConfig(seq_dim=X_seq.shape[1], amr_classes=Y_amr.shape[1], taxonomy_size=len(set(taxonomy)) if taxonomy is not None else 0, use_taxonomy=args.use_taxonomy)
    model = mutils.HyperAMR(mcfg)

    class_weights = mutils.compute_class_weights(Y_amr[split == "train"]) if args.class_weights else None
    tcfg = mutils.TrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, lambda_align=args.lambda_align, lambda_bce=args.lambda_bce, lambda_tax=args.lambda_tax)

    state = mutils.train_model(model, train_loader, val_loader, tcfg, class_weights=class_weights)
    results = mutils.evaluate(model, test_loader, class_list)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch_path = artifacts_dir / "model.pt"
    torch.save({"model_state": model.state_dict(), "config": mcfg.__dict__}, torch_path)
    np.savez(artifacts_dir / "predictions.npz", logits=results.logits, targets=results.targets)
    print(f"Saved model -> {torch_path}")
    print(f"Test macro AUPR={results.macro_aupr:.4f} | AUROC={results.macro_auroc:.4f}")



def cmd_plot(args: argparse.Namespace):
    artifacts_dir = Path(args.artifacts)
    preds = np.load(artifacts_dir / "predictions.npz")
    with open(artifacts_dir / "amr_class_list.json") as f:
        class_list = json.load(f)["class_list"]

    logits = preds["logits"]
    targets = preds["targets"]

    freq = plotting.plot_class_frequency(targets, class_list, output_path=artifacts_dir / "figures" / "class_frequency.png")
    pr_summary = plotting.plot_pr_curves(logits, targets, class_list, output_path=artifacts_dir / "figures" / "pr_curves.png")

    print(freq.head())
    print(pr_summary)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hyperbolic AMR pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    dlp = sub.add_parser("download", help="Download genomes from URLs")
    dlp.add_argument("--urls", required=True, help="Comma-delimited list of genome URLs")
    dlp.add_argument("--output", required=True, help="Output directory for FASTA files")
    dlp.set_defaults(func=cmd_download)

    amr = sub.add_parser("run-amrfinder", help="Run AMRFinderPlus on a FASTA file")
    amr.add_argument("--fasta", required=True)
    amr.add_argument("--output", required=True)
    amr.add_argument("--threads", type=int, default=4)
    amr.add_argument("--database", default=None, help="Optional AMRFinder database path")
    amr.add_argument(
        "--run-many",
        action="store_true",
        help="Treat --fasta as a comma-delimited list and run AMRFinder on each entry",
    )
    amr.set_defaults(func=cmd_run_amrfinder)

    prep = sub.add_parser("prepare", help="Build contig labels and attach sequences")
    prep.add_argument("--tsvs", required=True, help="Comma-delimited AMRFinder TSV paths")
    prep.add_argument("--fastas", required=True, help="Comma-delimited FASTA paths")
    prep.add_argument("--output", required=True, help="Artifact output directory")
    prep.set_defaults(func=cmd_prepare)

    tr = sub.add_parser("train", help="Hash sequences and train the hyperbolic AMR model")
    tr.add_argument("--artifacts", required=True, help="Directory with contig_amr_labels.parquet/amr_class_list.json")
    tr.add_argument("--fastas", required=True, help="Comma-delimited FASTA paths for sequences")
    tr.add_argument("--k", type=int, default=5)
    tr.add_argument("--buckets", type=int, default=4096)
    tr.add_argument("--stride", type=int, default=1)
    tr.add_argument("--max-len", dest="max_len", type=int, default=None)
    tr.add_argument("--hash-threads", dest="hash_threads", type=int, default=1, help="Threads for k-mer hashing")
    tr.add_argument(
        "--loader-workers",
        dest="loader_workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers for training/eval",
    )
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4)
    tr.add_argument("--lambda-align", dest="lambda_align", type=float, default=1.0)
    tr.add_argument("--lambda-bce", dest="lambda_bce", type=float, default=1.0)
    tr.add_argument("--lambda-tax", dest="lambda_tax", type=float, default=0.0)
    tr.add_argument("--use-taxonomy", action="store_true")
    tr.add_argument("--class-weights", dest="class_weights", action="store_true", help="Use BCE positive weights")
    tr.set_defaults(func=cmd_train)

    pl = sub.add_parser("plot", help="Plot AMR class frequency and PR curves from saved predictions")
    pl.add_argument("--artifacts", required=True, help="Directory containing predictions.npz and amr_class_list.json")
    pl.set_defaults(func=cmd_plot)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

