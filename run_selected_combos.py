from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
MODULES = ["soft_hg", "dominant_distill", "discrim_elastic"]

DATASET_CONFIGS = {
    "cora": "configs/node/cora.yaml",
    "wisconsin": "configs/node/wisconsin.yaml",
    "cornell": "configs/node/cornell.yaml",
    "actor": "configs/node/actor.yaml",
    "chameleon": "configs/node/chameleon.yaml",
    "chameleon_filtered": "configs/node/chameleon_filtered.yaml",
    "chameleon_filtered_directed": "configs/node/chameleon_filtered_directed.yaml",
    "squirrel_filtered": "configs/node/squirrel_filtered.yaml",
    "squirrel_filtered_directed": "configs/node/squirrel_filtered_directed.yaml",
    "texas": "configs/node/texas.yaml",
    "penn94": "configs/node/penn94.yaml",
    "reed98": "configs/node/reed98.yaml",
    "amherst41": "configs/node/amherst41.yaml",
    "cornell5": "configs/node/cornell5.yaml",
    "johnshopkins55": "configs/node/johnshopkins55.yaml",
    "tolokers": "configs/node/tolokers.yaml",
    "minesweeper": "configs/node/minesweeper.yaml",
}

DEFAULT_DATASETS = ["chameleon_filtered", "squirrel_filtered", "texas", "tolokers"]

RECOMMENDED_VARIANTS = {
    "cora": ["ce_only"],
    "wisconsin": ["dominant_distill+discrim_elastic"],
    "cornell": ["dominant_distill+discrim_elastic"],
    "actor": ["dominant_distill"],
    "chameleon": ["soft_hg+discrim_elastic"],
    "chameleon_filtered": ["soft_hg+discrim_elastic", "ce_only", "dominant_distill"],
    "chameleon_filtered_directed": ["ce_only", "dominant_distill"],
    "squirrel_filtered": ["soft_hg+discrim_elastic", "ce_only", "dominant_distill"],
    "squirrel_filtered_directed": ["ce_only", "dominant_distill"],
    "texas": ["dominant_distill+discrim_elastic", "soft_hg+discrim_elastic", "ce_only"],
    "penn94": ["ce_only", "dominant_distill", "dominant_distill+discrim_elastic"],
    "reed98": ["ce_only", "dominant_distill", "dominant_distill+discrim_elastic"],
    "amherst41": ["ce_only", "dominant_distill", "dominant_distill+discrim_elastic"],
    "cornell5": ["ce_only", "dominant_distill", "dominant_distill+discrim_elastic"],
    "johnshopkins55": ["ce_only", "dominant_distill", "dominant_distill+discrim_elastic"],
    "tolokers": ["ce_only", "dominant_distill"],
    "minesweeper": ["ce_only", "dominant_distill"],
}


def all_variants() -> list[str]:
    variants = ["ce_only", "baseline"]
    for r in range(1, len(MODULES) + 1):
        for combo in itertools.combinations(MODULES, r):
            variants.append("+".join(combo))
    return variants


def run_cmd(cmd: list[str]) -> None:
    print("\n" + "=" * 100)
    print("RUN:", " ".join(cmd))
    print("=" * 100)
    subprocess.check_call(cmd, cwd=ROOT)


def dataset_variants(dataset_names: list[str], explicit_variants: list[str] | None, run_all_variants: bool):
    selected = {}
    for name in dataset_names:
        if explicit_variants:
            selected[name] = explicit_variants
        elif run_all_variants:
            selected[name] = all_variants()
        else:
            selected[name] = RECOMMENDED_VARIANTS[name]
    return selected


def aggregate(output_root: Path, selected: dict[str, list[str]]) -> None:
    def fmt_num(value):
        return "" if value is None else f"{value:.4f}"

    rows = []
    for dataset_name, variants in selected.items():
        for variant in variants:
            summary_path = output_root / dataset_name / variant / "summary.json"
            if not summary_path.exists():
                continue
            obj = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "dataset": dataset_name,
                    "variant": variant,
                    "metric": obj.get("metric", "accuracy"),
                    "best_val_mean": obj.get("best_val_mean"),
                    "best_test_mean": obj.get("best_test_mean"),
                    "best_test_std": obj.get("best_test_std"),
                    "best_single_test": obj.get("best_single_test"),
                    "best_single_seed": obj.get("best_single_seed"),
                }
            )
    rows.sort(key=lambda x: (x["dataset"], -(x["best_test_mean"] or -1.0)))
    (output_root / "aggregate.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_lines = [
        "dataset,variant,metric,best_val_mean,best_test_mean,best_test_std,best_single_test,best_single_seed"
    ]
    for row in rows:
        csv_lines.append(
            ",".join(
                [
                    row["dataset"],
                    row["variant"],
                    str(row["metric"]),
                    str(row["best_val_mean"]),
                    str(row["best_test_mean"]),
                    str(row["best_test_std"]),
                    str(row["best_single_test"]),
                    str(row["best_single_seed"]),
                ]
            )
        )
    (output_root / "aggregate.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    md = [
        "# Curated Node Search Results",
        "",
        "| dataset | variant | metric | best_val_mean | best_test_mean | best_test_std | best_single_test | best_single_seed |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['dataset']} | {row['variant']} | {row['metric']} | "
            f"{fmt_num(row['best_val_mean'])} | {fmt_num(row['best_test_mean'])} | {fmt_num(row['best_test_std'])} | "
            f"{fmt_num(row['best_single_test'])} | {row['best_single_seed']} |"
        )
    (output_root / "aggregate.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a curated set of node-classification experiments.")
    ap.add_argument("--skip-bootstrap", action="store_true", help="Skip dataset downloads.")
    ap.add_argument("--output-root", default="combo_runs/search", help="Output directory for current search runs.")
    ap.add_argument("--all-datasets", action="store_true", help="Run every dataset registered in this file.")
    ap.add_argument("--all-variants", action="store_true", help="Run the exhaustive variant list for each dataset.")
    ap.add_argument("--variants", nargs="*", help="Override variants for every selected dataset.")
    ap.add_argument("--datasets", nargs="*", help="Datasets to run. Defaults to the active search pool.")
    ap.add_argument("--seeds", nargs="*", type=int, help="Override seeds from config files.")
    args = ap.parse_args()

    dataset_names = args.datasets or (list(DATASET_CONFIGS) if args.all_datasets else DEFAULT_DATASETS)
    unknown = sorted(set(dataset_names) - set(DATASET_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    selected = dataset_variants(dataset_names, args.variants, args.all_variants)

    if not args.skip_bootstrap:
        run_cmd([PYTHON, str(ROOT / "scripts" / "bootstrap_datasets.py"), "--group", "all"])

    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    for dataset_name in dataset_names:
        cfg_rel = DATASET_CONFIGS[dataset_name]
        for variant in selected[dataset_name]:
            cmd = [
                PYTHON,
                str(ROOT / "run_node.py"),
                "--config",
                str(ROOT / cfg_rel),
                "--output-dir",
                str(output_root),
                "--variant",
                variant,
            ]
            if args.seeds:
                cmd.extend(["--seeds", *[str(seed) for seed in args.seeds]])
            run_cmd(cmd)

    aggregate(output_root, selected)
    print("\nCurated node search finished. Results:", output_root)


if __name__ == "__main__":
    main()
