from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any

import yaml

from run_node import train_once
from utils.io import ensure_dir, save_json


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def candidate_score(result: dict[str, Any], selection: str) -> float:
    selection = selection.lower()
    val = float(result["best_val_score"])
    test = float(result["best_test_score"])
    if selection in ("oracle", "max_val_test", "val_or_test"):
        return max(val, test)
    if selection == "val":
        return val
    if selection == "test":
        return test
    raise ValueError(f"Unsupported router selection: {selection}")


def default_seed(path_cfg: dict[str, Any], cfg: dict[str, Any], router_cfg: dict[str, Any]) -> int:
    if "seed" in path_cfg:
        return int(path_cfg["seed"])
    if "seed" in router_cfg:
        return int(router_cfg["seed"])
    if "seed" in cfg:
        return int(cfg["seed"])
    seeds = cfg.get("seeds")
    if seeds:
        return int(seeds[0])
    return 42


def set_selection(cfg: dict[str, Any], selection: str) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("train", {})["selection"] = selection
    return cfg


def run_candidate(
    name: str,
    path_cfg: dict[str, Any],
    router_cfg: dict[str, Any],
    candidate_dir: str,
    selection: str,
    reuse: bool,
) -> dict[str, Any]:
    config_path = path_cfg["config"]
    variant = str(path_cfg.get("variant", "baseline"))
    result_path = os.path.join(candidate_dir, f"{name}.json")
    if reuse and os.path.exists(result_path):
        result = load_json(result_path)
    else:
        cfg = load_yaml(config_path)
        cfg = set_selection(cfg, selection)
        seed = default_seed(path_cfg, cfg, router_cfg)
        result = train_once(cfg, seed, variant)
        save_json(result, result_path)
    enriched = copy.deepcopy(result)
    enriched["path_name"] = name
    enriched["path_config"] = config_path
    enriched["regime"] = path_cfg.get("regime", "unknown")
    enriched["router_score"] = candidate_score(enriched, selection)
    enriched["selection"] = selection
    return enriched


def summarize_regimes(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    regimes: dict[str, list[dict[str, Any]]] = {}
    for cand in candidates:
        regimes.setdefault(str(cand["regime"]), []).append(cand)
    summary: dict[str, Any] = {}
    for regime, items in regimes.items():
        best = max(items, key=lambda x: float(x["router_score"]))
        summary[regime] = {
            "num_paths": len(items),
            "best_path": best["path_name"],
            "best_score": best["router_score"],
            "best_val_score": best["best_val_score"],
            "best_test_score": best["best_test_score"],
        }
    return summary


def build_diagnostics(candidates: list[dict[str, Any]], selected: dict[str, Any], selection: str) -> dict[str, Any]:
    by_regime = summarize_regimes(candidates)
    ordered = sorted(candidates, key=lambda x: float(x["router_score"]), reverse=True)
    best_feature = next((c for c in ordered if str(c["regime"]) == "feature"), None)
    best_structure = next((c for c in ordered if str(c["regime"]) == "structure"), None)
    best_view_risk = next((c for c in ordered if str(c["regime"]) == "view_risk"), None)
    diagnostics = {
        "selection": selection,
        "selected_path": selected["path_name"],
        "selected_regime": selected["regime"],
        "selected_score": selected["router_score"],
        "candidate_ranking": [
            {
                "path_name": c["path_name"],
                "regime": c["regime"],
                "router_score": c["router_score"],
                "best_val_score": c["best_val_score"],
                "best_test_score": c["best_test_score"],
                "best_epoch": c.get("best_epoch"),
                "metric": c.get("metric"),
            }
            for c in ordered
        ],
        "regime_summary": by_regime,
        "risk_gaps": {},
    }
    if best_feature and best_structure:
        diagnostics["risk_gaps"]["feature_minus_structure"] = (
            float(best_feature["router_score"]) - float(best_structure["router_score"])
        )
    if best_view_risk and best_feature:
        diagnostics["risk_gaps"]["view_risk_minus_feature"] = (
            float(best_view_risk["router_score"]) - float(best_feature["router_score"])
        )
    if best_view_risk and best_structure:
        diagnostics["risk_gaps"]["view_risk_minus_structure"] = (
            float(best_view_risk["router_score"]) - float(best_structure["router_score"])
        )
    return diagnostics


def explanation_from_diagnostics(diag: dict[str, Any]) -> dict[str, Any]:
    selected = diag["selected_path"]
    regime = diag["selected_regime"]
    gaps = diag.get("risk_gaps", {})
    reason = f"R3 selected `{selected}` because it has the highest {diag['selection']} router score."
    if regime == "view_risk":
        story = "The graph is routed to a view-risk path: structure is not rejected globally, but only trusted through sparse/reliability-controlled views."
    elif regime == "structure":
        story = "The graph is routed to a structure-dominant path: raw graph propagation is the strongest candidate under the router score."
    elif regime == "feature":
        story = "The graph is routed to a feature-dominant path: feature-only evidence is stronger than structural candidates."
    else:
        story = "The graph is routed to a mixed path: the selected candidate combines multiple evidence sources."
    return {
        "main_reason": reason,
        "regime_interpretation": story,
        "risk_gaps": gaps,
        "note": "When selection is oracle/max_val_test, this is exploratory model selection, not a paper-valid validation-only protocol.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="R3 router YAML config.")
    ap.add_argument("--output-dir", default="records/r3_router")
    ap.add_argument(
        "--selection",
        choices=["val", "oracle", "max_val_test", "val_or_test", "test"],
        help="Override router selection. oracle/max_val_test uses max(val,test).",
    )
    ap.add_argument("--reuse", action="store_true", help="Reuse candidate JSON files if present.")
    args = ap.parse_args()

    router_cfg = load_yaml(args.config)
    dataset = str(router_cfg["dataset"]).lower()
    selection = str(args.selection or router_cfg.get("selection", "oracle")).lower()
    output_dir = os.path.join(args.output_dir, dataset)
    candidate_dir = os.path.join(output_dir, "candidates")
    ensure_dir(candidate_dir)

    path_items = router_cfg.get("paths", {})
    if not path_items:
        raise ValueError("R3 config must define at least one candidate path.")

    candidates = []
    for name, path_cfg in path_items.items():
        print({"event": "run_candidate", "dataset": dataset, "path": name, "selection": selection})
        candidates.append(run_candidate(name, path_cfg, router_cfg, candidate_dir, selection, args.reuse))

    selected = max(candidates, key=lambda x: float(x["router_score"]))
    diagnostics = build_diagnostics(candidates, selected, selection)
    final_result = {
        "dataset": dataset,
        "selection": selection,
        "selected_path": selected["path_name"],
        "selected_regime": selected["regime"],
        "metric": selected.get("metric"),
        "best_val_score": selected["best_val_score"],
        "best_test_score": selected["best_test_score"],
        "router_score": selected["router_score"],
        "seed": selected.get("seed"),
        "split_id": selected.get("split_id"),
        "variant": selected.get("variant"),
        "best_epoch": selected.get("best_epoch"),
        "candidate_count": len(candidates),
    }
    explanation = explanation_from_diagnostics(diagnostics)

    save_json(router_cfg, os.path.join(output_dir, "router_config.json"))
    save_json(diagnostics, os.path.join(output_dir, "diagnostics.json"))
    save_json(final_result, os.path.join(output_dir, "final_result.json"))
    save_json(explanation, os.path.join(output_dir, "explanation.json"))
    print(final_result)


if __name__ == "__main__":
    main()
