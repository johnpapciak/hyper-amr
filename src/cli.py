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


def cmd_subsample(args: argparse.Namespace):
    tsvs = [Path(p) for p in _json_list(args.tsvs)]
    fastas = [Path(p) for p in _json_list(args.fastas)]
    if not tsvs or not fastas:
        raise ValueError("Provide at least one TSV and FASTA to subsample")

    tsv_by_stem: dict[str, Path] = {}
    for t in tsvs:
        stem = Path(t).stem.replace(".amrfinder", "")
        if stem in tsv_by_stem:
            raise ValueError(f"Duplicate TSV stem detected: {stem}")
        tsv_by_stem[stem] = t

    out_dir = Path(args.output)
    for fasta in fastas:
        stem = Path(fasta).stem
        tsv_path = tsv_by_stem.get(stem)
        if tsv_path is None:
            raise ValueError(f"No TSV found for FASTA stem '{stem}'")

        fasta_out, tsv_out = dutils.subsample_fasta_and_tsv(
            fasta,
            tsv_path,
            out_dir,
            frac=args.frac,
            n_contigs=args.num_contigs,
            max_contigs=args.max_contigs,
            seed=args.seed,
        )
        print(f"Subsampled FASTA -> {fasta_out}")
        print(f"Subsampled TSV   -> {tsv_out}")


def cmd_prepare(args: argparse.Namespace):
    tsvs = [Path(p) for p in _json_list(args.tsvs)]
    fastas = [Path(p) for p in _json_list(args.fastas)]
    out_dir = Path(args.output)
    taxonomy_map = Path(args.taxonomy_map) if args.taxonomy_map else None
    lineage_path = Path(args.taxonomy_lineages) if args.taxonomy_lineages else None

    artifacts = dutils.build_amr_labels(tsvs, fastas, out_dir)
    df = pd.read_parquet(artifacts.labels_path)
    if taxonomy_map is not None:
        df = dutils.attach_taxonomy(df, taxonomy_map, lineage_path)
    df = dutils.attach_sequences(df, fastas)
    df = dutils.train_val_test_split(df)
    df.to_parquet(artifacts.labels_path, index=False)
    print(f"Saved labels -> {artifacts.labels_path}")
    print(f"Saved classes -> {artifacts.classes_path}")


def cmd_train(args: argparse.Namespace):
    artifacts_dir = Path(args.artifacts)
    labels_path = artifacts_dir / "contig_amr_labels.parquet"
    classes_path = artifacts_dir / "amr_class_list.json"
    output_suffix = args.output_suffix.strip()

    def _with_suffix(path: Path) -> Path:
        return path.with_name(f"{path.stem}_{output_suffix}{path.suffix}") if output_suffix else path

    df = pd.read_parquet(labels_path)
    with open(classes_path) as f:
        class_list = json.load(f)["class_list"]

    cfg = dutils.HashingConfig(k=args.k, buckets=args.buckets, stride=args.stride, max_len=args.max_len)
    df = dutils.attach_sequences(df, [Path(p) for p in _json_list(args.fastas)])
    X_seq, Y_amr = dutils.prepare_features(df, class_list, cfg, threads=args.hash_threads)

    split = df["split"].values if "split" in df.columns else dutils.train_val_test_split(df)["split"].values
    taxonomy_cols = _json_list(args.taxonomy_cols) if args.taxonomy_cols else []
    taxonomy = None
    taxonomy_levels = 1
    taxid2idx: dict[str, int] = {}
    if args.use_taxonomy:
        lineage_pref_order = [
            "domain",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ]
        if taxonomy_cols:
            cols = taxonomy_cols
        else:
            detected_lineage = [c for c in lineage_pref_order if c in df.columns]
            cols = detected_lineage if detected_lineage else (["taxid"] if "taxid" in df.columns else [])

        taxonomy, taxid2idx = dutils.encode_taxonomy_lineage(df, cols)
        if taxonomy is not None:
            taxonomy_levels = taxonomy.shape[1]

    train_loader, val_loader, test_loader = mutils.build_dataloaders(
        X_seq, Y_amr, split, taxonomy, num_workers=args.loader_workers
    )

    taxonomy_size = (len(taxid2idx) + 1) if taxid2idx else 0
    mcfg = mutils.ModelConfig(
        seq_dim=X_seq.shape[1],
        amr_classes=Y_amr.shape[1],
        taxonomy_size=taxonomy_size,
        use_taxonomy=args.use_taxonomy,
        taxonomy_levels=taxonomy_levels,
    )
    model = mutils.HyperAMR(mcfg)

    class_weights = mutils.compute_class_weights(Y_amr[split == "train"]) if args.class_weights else None
    tcfg = mutils.TrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, lambda_align=args.lambda_align, lambda_bce=args.lambda_bce, lambda_tax=args.lambda_tax)

    state = mutils.train_model(model, train_loader, val_loader, tcfg, class_weights=class_weights)
    results = mutils.evaluate(
        model,
        test_loader,
        class_list,
        calibrate_thresholds=args.calibrate_thresholds,
        calibration_loader=val_loader if args.calibrate_thresholds else None,
        calibration_metric=args.calibration_metric,
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch_path = _with_suffix(artifacts_dir / "model.pt")
    torch.save({"model_state": model.state_dict(), "config": mcfg.__dict__}, torch_path)
    np.savez(
        _with_suffix(artifacts_dir / "predictions.npz"),
        logits=results.logits,
        targets=results.targets,
        predictions=results.predictions,
        thresholds=results.thresholds,
    )
    metrics_df = pd.DataFrame(
        {
            "amr_class": class_list,
            "accuracy": results.accuracy_per_class,
            "aupr": results.aupr_per_class,
            "auroc": results.auroc_per_class,
        }
    )
    metrics_df = pd.concat(
        [
            metrics_df,
            pd.DataFrame(
                [
                    {
                        "amr_class": "OVERALL",
                        "accuracy": results.overall_accuracy,
                        "aupr": results.macro_aupr,
                        "auroc": results.macro_auroc,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    metrics_path = _with_suffix(artifacts_dir / "evaluation_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved model -> {torch_path}")
    print(f"Saved evaluation metrics -> {metrics_path}")
    print(f"Test macro AUPR={results.macro_aupr:.4f} | AUROC={results.macro_auroc:.4f}")
 

def cmd_balance(args: argparse.Namespace):
    artifacts_dir = Path(args.artifacts)
    labels_path = artifacts_dir / "contig_amr_labels.parquet"
    classes_path = artifacts_dir / "amr_class_list.json"

    df = pd.read_parquet(labels_path)
    with open(classes_path) as f:
        class_list = json.load(f)["class_list"]

    fastas = [Path(p) for p in _json_list(args.fastas)]
    if not fastas:
        raise ValueError("Provide at least one FASTA path with --fastas")

    needs_source = "source_file" not in df.columns or df["source_file"].isna().any()
    if needs_source:
        contig_to_source: dict[str, str] = {}
        for fasta in fastas:
            src = fasta.stem
            for cid in dutils.all_contig_ids_from_fasta(fasta):
                contig_to_source[cid] = src
                contig_to_source[dutils.right_of_pipe(cid)] = src

        if "source_file" not in df.columns:
            df["source_file"] = df["contig_id"].map(contig_to_source)
        else:
            missing = df["source_file"].isna()
            df.loc[missing, "source_file"] = df.loc[missing, "contig_id"].map(contig_to_source)

        if df["source_file"].isna().any():
            missing_ids = df[df["source_file"].isna()]["contig_id"].unique().tolist()
            raise ValueError(
                "Unable to infer source_file for contigs: "
                + ", ".join(missing_ids[:5])
                + ("..." if len(missing_ids) > 5 else "")
            )

    balance = dutils.amr_balance_by_source(df, class_list)
    out_path = artifacts_dir / "label_balance_by_source.csv"
    balance.to_csv(out_path, index=False)

    print(balance.head())
    print(f"Saved label balance -> {out_path}")


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

    sub_sample = sub.add_parser(
        "subsample", help="Subsample FASTA/TSV pairs without rerunning AMRFinder"
    )
    sub_sample.add_argument("--tsvs", required=True, help="Comma-delimited AMRFinder TSV paths")
    sub_sample.add_argument("--fastas", required=True, help="Comma-delimited FASTA paths")
    sub_sample.add_argument("--output", required=True, help="Output directory for subsampled files")
    sub_sample.add_argument(
        "--frac",
        type=float,
        default=None,
        help="Fraction of contigs to keep (provide this or --num-contigs)",
    )
    sub_sample.add_argument(
        "--num-contigs",
        dest="num_contigs",
        type=int,
        default=None,
        help="Absolute number of contigs to keep (provide this or --frac)",
    )
    sub_sample.add_argument(
        "--max-contigs",
        dest="max_contigs",
        type=int,
        default=None,
        help=(
            "Optional cap on contigs kept per file (applied after --frac/--num-contigs); "
            "use with --frac 1.0 to only trim oversized FASTAs while leaving smaller ones intact"
        ),
    )
    sub_sample.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    sub_sample.set_defaults(func=cmd_subsample)

    prep = sub.add_parser("prepare", help="Build contig labels and attach sequences")
    prep.add_argument("--tsvs", required=True, help="Comma-delimited AMRFinder TSV paths")
    prep.add_argument("--fastas", required=True, help="Comma-delimited FASTA paths")
    prep.add_argument("--output", required=True, help="Artifact output directory")
    prep.add_argument(
        "--taxonomy-map",
        default=None,
        help="Optional TSV mapping contigs/source_files to taxid (and lineage columns)",
    )
    prep.add_argument(
        "--taxonomy-lineages",
        default=None,
        help="Optional TSV with taxid and lineage ranks (domain,phylum,class,order,family,genus,species)",
    )
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
    tr.add_argument(
        "--output-suffix",
        dest="output_suffix",
        default="",
        help="Optional suffix to append to saved model/prediction/metrics outputs",
    )
    tr.add_argument(
        "--taxonomy-cols",
        default=None,
        help="Comma-delimited lineage columns (e.g., phylum,class,order,family,genus,species)",
    )
    tr.add_argument("--class-weights", dest="class_weights", action="store_true", help="Use BCE positive weights")
    tr.add_argument(
        "--calibrate-thresholds",
        action="store_true",
        help="Search per-class decision thresholds on the validation set",
    )
    tr.add_argument(
        "--calibration-metric",
        default="f1",
        choices=["f1", "average_precision"],
        help="Objective to optimize during threshold search",
    )
    tr.set_defaults(func=cmd_train)

    bal = sub.add_parser(
        "balance",
        help="Summarize AMR label balance by source file (before training)",
    )
    bal.add_argument("--artifacts", required=True, help="Directory with contig_amr_labels.parquet")
    bal.add_argument(
        "--fastas",
        required=True,
        help="Comma-delimited FASTA paths (used to recover missing source_file entries)",
    )
    bal.set_defaults(func=cmd_balance)

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

