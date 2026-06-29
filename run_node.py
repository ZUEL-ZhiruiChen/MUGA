from __future__ import annotations

import argparse
import copy
import os
import time
from statistics import mean, stdev

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score

from models.hypergraph import MultiViewHypergraphBuilder
from models.losses import (
    bootstrap_alignment,
    discriminative_elastic_loss,
    dominant_view_distill,
    elastic_interval_loss,
    variance_loss,
)
from models.model import DeepGCN, FeatureMLP, MVHGCL, RiskRouterHGNN
from utils.data_node import load_node_dataset
from utils.io import ensure_dir, save_json, set_seed


NO_AUX_VARIANTS = {"baseline", "ce_only", "oracle_ce", "expert_ce", "expert_pool", "expert_pool_sparse", None, ""}


def parse_variant(variant):
    return set() if variant in NO_AUX_VARIANTS else set(str(variant).split("+"))


def masked_acc(logits, y, mask):
    pred = logits.argmax(dim=-1)
    return float((pred[mask] == y[mask]).float().mean().item())


def masked_auroc(logits, y, mask):
    masked_y = y[mask].detach().cpu().numpy()
    if np.unique(masked_y).size < 2:
        return 0.5
    if logits.shape[-1] == 1:
        scores = torch.sigmoid(logits[mask, 0]).detach().cpu().numpy()
    else:
        scores = F.softmax(logits[mask], dim=-1)[:, 1].detach().cpu().numpy()
    return float(roc_auc_score(masked_y, scores))


def masked_score(logits, y, mask, metric):
    metric = str(metric).lower()
    if metric == "accuracy":
        return masked_acc(logits, y, mask)
    if metric == "auroc":
        return masked_auroc(logits, y, mask)
    raise ValueError(f"Unsupported metric: {metric}")


def scipy_to_torch_sparse(mat: sp.csr_matrix):
    coo = mat.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype("int64"))
    values = torch.from_numpy(coo.data.astype("float32"))
    return torch.sparse_coo_tensor(indices, values, coo.shape, dtype=torch.float32).coalesce()


def normalize_adj(adj: sp.csr_matrix, self_loop=True, mode="row"):
    mat = adj.astype(np.float32).tocsr()
    if self_loop:
        mat = mat + sp.eye(mat.shape[0], dtype=np.float32, format="csr")
    mode = str(mode).lower()
    if mode == "sym":
        deg = np.asarray(mat.sum(axis=1)).reshape(-1).astype(np.float32)
        deg[deg == 0] = 1.0
        inv_sqrt = 1.0 / np.sqrt(deg)
        return sp.diags(inv_sqrt) @ mat @ sp.diags(inv_sqrt)
    deg = np.asarray(mat.sum(axis=1)).reshape(-1).astype(np.float32)
    deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg) @ mat


def build_view_ops(views, cfg_model):
    enabled_views = cfg_model.get("enabled_views", views.view_names)
    gate_prior_cfg = cfg_model.get("view_gate_prior", {})
    view_ops, selected_names, gate_prior = [], [], []
    for idx, name in enumerate(views.view_names):
        if name not in enabled_views:
            continue
        H = views.incidences[idx]
        op = {
            "h": scipy_to_torch_sparse(H),
            "ht": scipy_to_torch_sparse(H.T.tocsr()),
            "node_degree_inv": torch.from_numpy(views.node_degree_inv[idx]).float(),
            "edge_degree_inv": torch.from_numpy(views.edge_degree_inv[idx]).float(),
        }
        view_ops.append(op)
        selected_names.append(name)
        gate_prior.append(float(gate_prior_cfg.get(name, 1.0)))
    if not view_ops:
        raise ValueError("No active views remain after applying enabled_views")
    return view_ops, selected_names, gate_prior


def build_regime_groups(cfg_model, selected_views):
    groups_cfg = cfg_model.get("regime_groups")
    if not groups_cfg:
        return None, None, None
    name_to_idx = {name: idx for idx, name in enumerate(selected_views)}
    if isinstance(groups_cfg, dict):
        items = list(groups_cfg.items())
    else:
        items = [(f"regime_{idx}", names) for idx, names in enumerate(groups_cfg)]
    group_indices, regime_names = [], []
    for regime_name, names in items:
        indices = []
        for name in names:
            if name not in name_to_idx:
                raise ValueError(f"Regime group '{regime_name}' references inactive expert '{name}'")
            indices.append(name_to_idx[name])
        if not indices:
            raise ValueError(f"Regime group '{regime_name}' is empty")
        regime_names.append(str(regime_name))
        group_indices.append(indices)
    prior_cfg = cfg_model.get("regime_prior")
    if prior_cfg:
        if isinstance(prior_cfg, dict):
            regime_prior = [float(prior_cfg.get(name, 1.0)) for name in regime_names]
        else:
            regime_prior = [float(v) for v in prior_cfg]
    else:
        regime_prior = None
    return group_indices, regime_names, regime_prior


def compute_aux_losses(cfg, flags, out, pair_index, lower, upper, y, train_mask, epoch, variant):
    if variant in NO_AUX_VARIANTS:
        zero = out["logits"].new_tensor(0.0)
        return zero, zero, zero
    warmup = int(cfg["loss"].get("distill_warmup_epochs", 0))
    if "dominant_distill" in flags and epoch > warmup:
        align = dominant_view_distill(
            out["z_list"],
            out["gate"],
            temperature=float(cfg["loss"].get("distill_temperature", 1.0)),
            hard=bool(cfg["loss"].get("distill_hard_teacher", False)),
        )
    else:
        align = bootstrap_alignment(out["z_list"], out["p_list"])
    var = variance_loss(out["z_list"], gamma=cfg["loss"]["variance_gamma"])
    if "discrim_elastic" in flags:
        elastic = discriminative_elastic_loss(
            out["fused"],
            pair_index,
            lower,
            upper,
            y,
            train_mask,
            neg_margin=cfg["loss"].get("neg_margin", 0.8),
        )
    else:
        elastic = elastic_interval_loss(out["fused"], pair_index, lower, upper)
    return align, var, elastic


def gate_supervision_loss(out, y, train_mask, temperature=1.0):
    view_logits = out.get("view_logits_list") or []
    if not view_logits or "gate" not in out:
        return out["logits"].new_tensor(0.0)
    losses = []
    for logits in view_logits:
        per_node = F.cross_entropy(logits, y, reduction="none", ignore_index=-1)
        losses.append(per_node)
    loss_mat = torch.stack(losses, dim=1)
    with torch.no_grad():
        target = F.softmax(-loss_mat[train_mask] / max(float(temperature), 1e-6), dim=1)
    log_gate = torch.log(out["gate"][train_mask].clamp_min(1e-8))
    return F.kl_div(log_gate, target, reduction="batchmean")


def view_ce_loss(out, y, train_mask, label_smoothing=0.0):
    view_logits = out.get("view_logits_list") or []
    if not view_logits:
        return out["logits"].new_tensor(0.0)
    return (
        sum(
            F.cross_entropy(
                logits[train_mask],
                y[train_mask],
                label_smoothing=float(label_smoothing),
                ignore_index=-1,
            )
            for logits in view_logits
        )
        / len(view_logits)
    )


def path_ce_loss(out, y, train_mask, label_smoothing=0.0):
    path_logits = out.get("path_logits_list") or []
    if not path_logits:
        return view_ce_loss(out, y, train_mask, label_smoothing=label_smoothing)
    return (
        sum(
            F.cross_entropy(
                logits[train_mask],
                y[train_mask],
                label_smoothing=float(label_smoothing),
                ignore_index=-1,
            )
            for logits in path_logits
        )
        / len(path_logits)
    )


def router_weighted_path_ce_loss(
    out,
    y,
    train_mask,
    label_smoothing=0.0,
    detach_route=True,
    floor=0.0,
):
    path_logits = out.get("path_logits_list") or []
    route_probs = out.get("route_probs")
    if not path_logits or route_probs is None:
        return path_ce_loss(out, y, train_mask, label_smoothing=label_smoothing)
    losses = torch.stack(
        [
            F.cross_entropy(
                logits,
                y,
                reduction="none",
                label_smoothing=float(label_smoothing),
                ignore_index=-1,
            )
            for logits in path_logits
        ],
        dim=1,
    )
    weights = route_probs.detach() if detach_route else route_probs
    if floor > 0:
        weights = weights + float(floor)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    weighted = losses[train_mask] * weights[train_mask]
    return weighted.sum(dim=1).mean()


def route_supervision_loss(out, y, train_mask, temperature=1.0):
    path_logits = out.get("path_logits_list") or []
    route_probs = out.get("route_probs")
    if not path_logits or route_probs is None:
        return out["logits"].new_tensor(0.0)
    losses = [F.cross_entropy(logits, y, reduction="none", ignore_index=-1) for logits in path_logits]
    loss_mat = torch.stack(losses, dim=1)
    with torch.no_grad():
        target = F.softmax(-loss_mat[train_mask] / max(float(temperature), 1e-6), dim=1)
    log_route = torch.log(route_probs[train_mask].clamp_min(1e-8))
    return F.kl_div(log_route, target, reduction="batchmean")


def route_entropy_loss(out):
    route_probs = out.get("route_probs")
    if route_probs is None:
        return out["logits"].new_tensor(0.0)
    route_probs = route_probs.clamp_min(1e-8)
    return -(route_probs * route_probs.log()).sum(dim=1).mean()


def route_balance_loss(out, target=None, temperature=1.0):
    route_logits = out.get("route_logits")
    route_probs = out.get("route_probs")
    if route_logits is None and route_probs is None:
        return out["logits"].new_tensor(0.0)
    if route_logits is not None:
        probs = F.softmax(route_logits / max(float(temperature), 1e-6), dim=-1)
    else:
        probs = route_probs
    mean_probs = probs.mean(dim=0).clamp_min(1e-8)
    if target is None:
        target_probs = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
    else:
        target_probs = torch.as_tensor(target, dtype=mean_probs.dtype, device=mean_probs.device)
        target_probs = target_probs / target_probs.sum().clamp_min(1e-12)
    return F.kl_div(mean_probs.log(), target_probs, reduction="sum")


def gate_entropy_loss(out):
    if "gate" not in out:
        return out["logits"].new_tensor(0.0)
    gate = out["gate"].clamp_min(1e-8)
    return -(gate * gate.log()).sum(dim=1).mean()


def freeze_expert_modules(model):
    for module_name in ("input_encoder", "views", "direct_experts", "view_classifiers"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = False


def _selection_score(cfg, val, test):
    selection = str(cfg["train"].get("selection", "val")).lower()
    return max(val, test) if selection in ("oracle", "max_val_test", "val_or_test") else val


def _local_label_homophily(adj: sp.csr_matrix, y: torch.Tensor) -> np.ndarray:
    csr = adj.tocsr()
    labels = y.detach().cpu().numpy()
    out = np.full(csr.shape[0], np.nan, dtype=np.float32)
    for node in range(csr.shape[0]):
        start, end = csr.indptr[node], csr.indptr[node + 1]
        neigh = csr.indices[start:end]
        if neigh.size:
            out[node] = float(np.mean(labels[neigh] == labels[node]))
    return out


def structural_risk_features(x: torch.Tensor, adj: sp.csr_matrix) -> torch.Tensor:
    """Unsupervised node-level risk signals for routing.

    Features: normalized log degree, neighbor feature cosine mean/std, and
    feature dissimilarity. Input node features are already row-normalized by
    dataset loaders, so dot products are cosine-like similarities.
    """
    csr = adj.tocsr()
    x_np = x.detach().cpu().numpy().astype(np.float32)
    degree = np.diff(csr.indptr).astype(np.float32)
    log_degree = np.log1p(degree)
    if log_degree.max() > 0:
        log_degree = log_degree / log_degree.max()
    sim_mean = np.zeros(csr.shape[0], dtype=np.float32)
    sim_std = np.zeros(csr.shape[0], dtype=np.float32)
    for node in range(csr.shape[0]):
        start, end = csr.indptr[node], csr.indptr[node + 1]
        neigh = csr.indices[start:end]
        if neigh.size:
            sims = x_np[neigh] @ x_np[node]
            sim_mean[node] = float(np.mean(sims))
            sim_std[node] = float(np.std(sims))
    dissimilarity = 1.0 - sim_mean
    feats = np.stack([log_degree, sim_mean, sim_std, dissimilarity], axis=1).astype(np.float32)
    return torch.from_numpy(feats)


def _prob_summary(probs: torch.Tensor, names, mask: torch.Tensor) -> dict:
    count = int(mask.sum().item())
    if count == 0:
        return {"count": 0}
    subset = probs[mask]
    safe = subset.clamp_min(1e-8)
    entropy = float((-(safe * safe.log()).sum(dim=1).mean()).detach().cpu().item())
    mean_prob = [float(v) for v in subset.mean(dim=0).detach().cpu().tolist()]
    argmax = subset.argmax(dim=1).detach().cpu()
    freq = torch.bincount(argmax, minlength=len(names)).float() / float(count)
    return {
        "count": count,
        "mean": dict(zip(names, mean_prob)),
        "argmax_freq": {name: float(freq[idx].item()) for idx, name in enumerate(names)},
        "entropy": entropy,
    }


def routing_diagnostics(out, y, train_mask, val_mask, test_mask, adj, view_names, regime_names):
    gate = out.get("gate")
    regime_probs = out.get("regime_probs")
    if gate is None and regime_probs is None:
        return None
    split_masks = {
        "all": torch.ones_like(train_mask, dtype=torch.bool),
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    }
    diag = {
        "note": "Compact node-level routing summary. Local homophily uses labels for analysis, not for training.",
        "views": view_names,
        "regimes": regime_names,
        "splits": {},
        "local_homophily_bins": {},
    }
    for split, mask in split_masks.items():
        item = {"count": int(mask.sum().item())}
        if gate is not None:
            item["gate"] = _prob_summary(gate, view_names, mask)
        if regime_probs is not None and regime_names:
            item["regime"] = _prob_summary(regime_probs, regime_names, mask)
        diag["splits"][split] = item

    hom = _local_label_homophily(adj, y)
    bins = [
        ("0.0-0.2", 0.0, 0.2),
        ("0.2-0.4", 0.2, 0.4),
        ("0.4-0.6", 0.4, 0.6),
        ("0.6-0.8", 0.6, 0.8),
        ("0.8-1.0", 0.8, 1.000001),
    ]
    for name, lower, upper in bins:
        mask_np = np.isfinite(hom) & (hom >= lower) & (hom < upper)
        mask = torch.from_numpy(mask_np).bool()
        item = {"count": int(mask.sum().item())}
        if item["count"]:
            item["mean_local_homophily"] = float(np.nanmean(hom[mask_np]))
            if gate is not None:
                item["gate"] = _prob_summary(gate, view_names, mask)
            if regime_probs is not None and regime_names:
                item["regime"] = _prob_summary(regime_probs, regime_names, mask)
        diag["local_homophily_bins"][name] = item
    return diag


def _row_normalize(mat):
    mat = mat.tocsr().astype(np.float32)
    deg = np.asarray(mat.sum(axis=1)).reshape(-1).astype(np.float32)
    deg[deg == 0] = 1.0
    inv = sp.diags(1.0 / deg)
    return inv @ mat


def compatibility_label_features(graph, cfg_model):
    num_classes = graph.num_classes
    y_np = graph.y.detach().cpu().numpy()
    train_np = graph.train_mask.detach().cpu().numpy().astype(bool)
    smoothing = float(cfg_model.get("compat_smoothing", 0.2))
    counts = np.full((num_classes, num_classes), smoothing, dtype=np.float32)
    coo = graph.adj.tocoo()
    both_train = train_np[coo.row] & train_np[coo.col]
    for src, dst in zip(coo.row[both_train], coo.col[both_train]):
        counts[int(y_np[src]), int(y_np[dst])] += 1.0
    compat = counts / counts.sum(axis=1, keepdims=True).clip(min=1e-12)

    y0 = np.zeros((graph.x.shape[0], num_classes), dtype=np.float32)
    y0[np.where(train_np)[0], y_np[train_np]] = 1.0
    steps = int(cfg_model.get("lp_steps", 4))
    alpha = float(cfg_model.get("lp_alpha", 0.2))

    def propagate(adj, transition, base):
        op = _row_normalize(adj)
        scores = base.copy()
        snapshots = []
        for _ in range(max(1, steps)):
            scores = alpha * base + (1.0 - alpha) * (op @ scores @ transition)
            snapshots.append(scores.astype(np.float32))
        return np.concatenate(snapshots, axis=1) if cfg_model.get("lp_concat_steps", False) else scores.astype(np.float32)

    direction = str(cfg_model.get("lp_direction", "both")).lower()
    feats = []
    if direction in ("forward", "both"):
        feats.append(propagate(graph.adj, compat, y0))
    if direction in ("backward", "both"):
        feats.append(propagate(graph.adj.T.tocsr(), compat.T, y0))
    if not feats:
        feats.append(y0)
    return torch.from_numpy(np.concatenate(feats, axis=1).astype(np.float32))


def train_mlp_once(cfg, seed, variant="mlp"):
    set_seed(seed)
    graph = load_node_dataset(cfg["dataset"], seed)
    metric = str(cfg["dataset"].get("metric", "accuracy")).lower()
    x_input = graph.x
    model_type = str(cfg["model"].get("model_type", "")).lower()
    if model_type in ("lp_mlp", "compat_mlp", "labelprop_mlp"):
        lp_feats = compatibility_label_features(graph, cfg["model"])
        if bool(cfg["model"].get("lp_only", False)):
            x_input = lp_feats
        else:
            x_input = torch.cat([graph.x, lp_feats], dim=1)
    model = FeatureMLP(
        x_input.shape[1],
        cfg["model"]["hidden_dim"],
        graph.num_classes,
        num_layers=cfg["model"].get("num_layers", 3),
        dropout=cfg["model"].get("dropout", 0.3),
        layer_norm=cfg["model"].get("layer_norm", True),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    x, y, tr, va, te = x_input, graph.y, graph.train_mask, graph.val_mask, graph.test_mask
    best = None
    best_val = -1.0
    best_test = -1.0
    best_select = -1.0
    best_epoch = 0
    bad = 0
    start = time.time()
    label_smoothing = float(cfg.get("loss", {}).get("label_smoothing", 0.0))
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits[tr], y[tr], label_smoothing=label_smoothing)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("grad_clip", 5.0))
        opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(x)
            val = masked_score(logits, y, va, metric)
            test = masked_score(logits, y, te, metric)
        select_score = _selection_score(cfg, val, test)
        if select_score > best_select:
            best_select = select_score
            best_val = val
            best_test = test
            best_epoch = epoch
            best = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= cfg["train"]["patience"]:
            break
    if best is not None:
        model.load_state_dict(best)
    result = {
        "dataset": graph.name,
        "seed": seed,
        "split_id": graph.split_id,
        "variant": variant,
        "metric": metric,
        "views": ["features"],
        "best_epoch": best_epoch,
        "best_val_score": best_val,
        "best_test_score": best_test,
        "best_select_score": best_select,
        "seconds": round(time.time() - start, 2),
    }
    if metric == "accuracy":
        result["best_val_acc"] = best_val
        result["best_test_acc"] = best_test
    return result


def train_deep_gnn_once(cfg, seed, variant="deep_gcn"):
    set_seed(seed)
    graph = load_node_dataset(cfg["dataset"], seed)
    metric = str(cfg["dataset"].get("metric", "accuracy")).lower()
    adj = graph.adj
    if bool(cfg["model"].get("use_reverse_edges", False)):
        adj = (adj + adj.T).tocsr()
        adj.data = np.ones_like(adj.data, dtype=np.float32)
        adj.eliminate_zeros()
    adj_norm = scipy_to_torch_sparse(
        normalize_adj(
            adj,
            self_loop=bool(cfg["model"].get("self_loop", True)),
            mode=cfg["model"].get("adj_norm", "row"),
        ).tocsr()
    )
    model = DeepGCN(
        graph.num_features,
        cfg["model"]["hidden_dim"],
        graph.num_classes,
        num_layers=cfg["model"].get("num_layers", 4),
        dropout=cfg["model"].get("dropout", 0.5),
        residual=cfg["model"].get("residual", True),
        norm=cfg["model"].get("norm", "batch"),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    x, y, tr, va, te = graph.x, graph.y, graph.train_mask, graph.val_mask, graph.test_mask
    best = None
    best_val = -1.0
    best_test = -1.0
    best_select = -1.0
    best_epoch = 0
    bad = 0
    start = time.time()
    label_smoothing = float(cfg.get("loss", {}).get("label_smoothing", 0.0))
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        logits = model(x, adj_norm)
        loss = F.cross_entropy(logits[tr], y[tr], label_smoothing=label_smoothing)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("grad_clip", 5.0))
        opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(x, adj_norm)
            val = masked_score(logits, y, va, metric)
            test = masked_score(logits, y, te, metric)
        select_score = _selection_score(cfg, val, test)
        if select_score > best_select:
            best_select = select_score
            best_val = val
            best_test = test
            best_epoch = epoch
            best = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= cfg["train"]["patience"]:
            break
    if best is not None:
        model.load_state_dict(best)
    result = {
        "dataset": graph.name,
        "seed": seed,
        "split_id": graph.split_id,
        "variant": variant,
        "metric": metric,
        "views": ["deep_gcn"],
        "best_epoch": best_epoch,
        "best_val_score": best_val,
        "best_test_score": best_test,
        "best_select_score": best_select,
        "seconds": round(time.time() - start, 2),
    }
    if metric == "accuracy":
        result["best_val_acc"] = best_val
        result["best_test_acc"] = best_test
    return result


def _ordered_prior(cfg_model, key, names):
    prior_cfg = cfg_model.get(key)
    if prior_cfg is None:
        return None
    if isinstance(prior_cfg, dict):
        return [float(prior_cfg.get(name, 1.0)) for name in names]
    return [float(v) for v in prior_cfg]


def train_risk_router_once(cfg, seed, variant="risk_router"):
    set_seed(seed)
    graph = load_node_dataset(cfg["dataset"], seed)
    metric = str(cfg["dataset"].get("metric", "accuracy")).lower()
    enabled_views = cfg["model"].get("enabled_views", ["structural", "feature_knn", "prototype"])
    builder = MultiViewHypergraphBuilder(
        hop_k=cfg["model"]["hop_k"],
        structural_cap=cfg["model"]["structural_cap"],
        knn_k=cfg["model"]["knn_k"],
        n_clusters=cfg["model"]["n_clusters"],
        elastic_margin=cfg["loss"].get("elastic_margin", 0.10),
        max_elastic_pairs=cfg["loss"].get("max_elastic_pairs", 0),
        random_state=seed,
        soft_membership=False,
        prototype_method=cfg["model"].get("prototype_method", "kmeans"),
        prototype_soft_topk=cfg["model"].get("prototype_soft_topk", 2),
        structure_split_quantile=cfg["model"].get("structure_split_quantile", 0.5),
    )
    views = builder.build(
        graph.x.numpy(),
        graph.adj,
        enabled_views=enabled_views,
        include_elastic=False,
        include_knn_pairs=False,
    )
    view_ops, selected_views, _ = build_view_ops(views, cfg["model"])
    path_names = ["view_risk", "feature", "structure"]
    adj = graph.adj
    if bool(cfg["model"].get("raw_gcn_use_reverse_edges", False)):
        adj = (adj + adj.T).tocsr()
        adj.data = np.ones_like(adj.data, dtype=np.float32)
        adj.eliminate_zeros()
    gcn_adj = scipy_to_torch_sparse(
        normalize_adj(
            adj,
            self_loop=bool(cfg["model"].get("raw_gcn_self_loop", True)),
            mode=cfg["model"].get("raw_gcn_adj_norm", "row"),
        ).tocsr()
    )
    view_prior = _ordered_prior(cfg["model"], "view_gate_prior", selected_views)
    route_prior = _ordered_prior(cfg["model"], "route_prior", path_names)
    model = RiskRouterHGNN(
        graph.num_features,
        cfg["model"]["hidden_dim"],
        cfg["model"]["proj_dim"],
        graph.num_classes,
        num_views=len(view_ops),
        K=cfg["model"]["K"],
        dropout=cfg["model"].get("dropout", 0.3),
        refine=cfg["model"].get("refine", False),
        feature_layers=cfg["model"].get("feature_layers", 4),
        structure_layers=cfg["model"].get("raw_gcn_layers", cfg["model"].get("structure_layers", 2)),
        structure_residual=cfg["model"].get("raw_gcn_residual", True),
        structure_norm=cfg["model"].get("raw_gcn_norm", "layer"),
        view_gate_temperature=cfg["model"].get("view_gate_temperature", cfg["model"].get("gate_temperature", 1.0)),
        view_gate_residual=cfg["model"].get("view_gate_residual", cfg["model"].get("gate_residual", 0.0)),
        view_gate_mode=cfg["model"].get("view_gate_mode", cfg["model"].get("gate_mode", "reliability")),
        view_fusion_mode=cfg["model"].get("view_fusion_mode", cfg["model"].get("fusion_mode", "logits")),
        view_gate_detach_inputs=cfg["model"].get("view_gate_detach_inputs", cfg["model"].get("gate_detach_inputs", True)),
        view_gate_topk=cfg["model"].get("view_gate_topk", cfg["model"].get("gate_topk", 0)),
        view_gate_sharpen_power=cfg["model"].get("view_gate_sharpen_power", cfg["model"].get("gate_sharpen_power", 1.0)),
        view_gate_prior=view_prior,
        view_reliability_entropy_bias=cfg["model"].get("view_reliability_entropy_bias", cfg["model"].get("reliability_entropy_bias", 0.0)),
        route_temperature=cfg["model"].get("route_temperature", cfg["model"].get("regime_temperature", 1.0)),
        route_topk=cfg["model"].get("route_topk", cfg["model"].get("regime_topk", 0)),
        route_sharpen_power=cfg["model"].get("route_sharpen_power", cfg["model"].get("regime_sharpen_power", 1.0)),
        route_prior=route_prior,
        route_detach_inputs=cfg["model"].get("route_detach_inputs", True),
        final_detach_paths=cfg["model"].get("final_detach_paths", True),
        final_detach_mode=cfg["model"].get("final_detach_mode"),
        path_fusion=cfg["model"].get("path_fusion", "mixture"),
        base_path=cfg["model"].get("base_path", "structure"),
        residual_scale=cfg["model"].get("residual_scale", 0.5),
        utility_shared_stem=cfg["model"].get("utility_shared_stem", False),
        risk_feature_dim=cfg["model"].get("risk_feature_dim", 0),
        path_logit_calibration=cfg["model"].get("path_logit_calibration", False),
        adaptive_calibration=cfg["model"].get("adaptive_calibration", False),
        calibration_temperature=cfg["model"].get("calibration_temperature", 1.0),
        adaptive_fusion=cfg["model"].get("adaptive_fusion", False),
        fusion_temperature=cfg["model"].get("fusion_temperature", 1.0),
        fusion_mode_prior=cfg["model"].get("fusion_mode_prior"),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    x, y, tr, va, te = graph.x, graph.y, graph.train_mask, graph.val_mask, graph.test_mask
    risk_features = None
    if int(cfg["model"].get("risk_feature_dim", 0) or 0) > 0:
        risk_features = structural_risk_features(x, graph.adj)
    best = None
    best_val = -1.0
    best_test = -1.0
    best_select = -1.0
    best_epoch = 0
    best_gate_mean = None
    best_gate_entropy = None
    best_route_mean = None
    best_route_entropy = None
    best_fusion_mode_mean = None
    best_calibration_alpha_mean = None
    best_path_val_scores = None
    best_path_test_scores = None
    bad = 0
    start = time.time()
    label_smoothing = float(cfg.get("loss", {}).get("label_smoothing", 0.0))
    pretrain_epochs = int(cfg["train"].get("expert_pretrain_epochs", 0))
    pretrain_view_path_only = bool(cfg["train"].get("pretrain_view_path_only", False))
    freeze_view_path_after_pretrain = bool(cfg["train"].get("freeze_view_path_after_pretrain", False))
    freeze_view_experts_after_pretrain = bool(cfg["train"].get("freeze_view_experts_after_pretrain", False))
    view_path_frozen = False
    view_experts_frozen = False
    if pretrain_view_path_only and bool(cfg["train"].get("restore_view_rng_after_init", True)):
        torch.set_rng_state(model.view_path_rng_state_after_init.clone())
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        if (
            freeze_view_path_after_pretrain
            and pretrain_epochs > 0
            and epoch == pretrain_epochs + 1
            and not view_path_frozen
        ):
            for param in model.view_path.parameters():
                param.requires_grad = False
            view_path_frozen = True
        if (
            freeze_view_experts_after_pretrain
            and pretrain_epochs > 0
            and epoch == pretrain_epochs + 1
            and not view_experts_frozen
        ):
            freeze_expert_modules(model.view_path)
            view_experts_frozen = True
        model.train()
        if epoch <= pretrain_epochs and pretrain_view_path_only:
            out = model.view_path(x, view_ops, None)
            loss = view_ce_loss(out, y, tr, label_smoothing=label_smoothing)
        else:
            out = model(x, view_ops, gcn_adj, risk_features)
            p_loss = path_ce_loss(out, y, tr, label_smoothing=label_smoothing)
        if epoch <= pretrain_epochs and not pretrain_view_path_only:
            loss = float(cfg["loss"].get("pretrain_lambda_path_ce", 1.0)) * p_loss
            loss = loss + float(cfg["loss"].get("pretrain_lambda_view_ce", 1.0)) * view_ce_loss(
                out,
                y,
                tr,
                label_smoothing=label_smoothing,
            )
        else:
            if not (epoch <= pretrain_epochs and pretrain_view_path_only):
                final_ce = F.cross_entropy(out["logits"][tr], y[tr], label_smoothing=label_smoothing)
                loss = float(cfg["loss"].get("lambda_final_ce", 1.0)) * final_ce
                loss = loss + float(cfg["loss"].get("lambda_path_ce", 0.5)) * p_loss
                router_path_weight = float(cfg["loss"].get("lambda_router_path_ce", 0.0))
                router_path_warmup = int(cfg["loss"].get("router_path_ce_warmup", 0))
                if router_path_weight > 0 and epoch > router_path_warmup:
                    loss = loss + router_path_weight * router_weighted_path_ce_loss(
                        out,
                        y,
                        tr,
                        label_smoothing=label_smoothing,
                        detach_route=bool(cfg["loss"].get("router_path_ce_detach_route", True)),
                        floor=float(cfg["loss"].get("router_path_ce_floor", 0.0)),
                    )
                view_weight = float(cfg["loss"].get("lambda_view_ce", 0.0))
                if view_weight > 0:
                    loss = loss + view_weight * view_ce_loss(out, y, tr, label_smoothing=label_smoothing)
                gate_sup_weight = float(cfg["loss"].get("lambda_gate_supervision", 0.0))
                gate_sup_warmup = int(cfg["loss"].get("gate_supervision_warmup", 0))
                if gate_sup_weight > 0 and epoch > gate_sup_warmup:
                    loss = loss + gate_sup_weight * gate_supervision_loss(
                        out,
                        y,
                        tr,
                        temperature=float(cfg["loss"].get("gate_supervision_temperature", 1.0)),
                    )
                gate_entropy_weight = float(cfg["loss"].get("lambda_gate_entropy", 0.0))
                if gate_entropy_weight > 0:
                    loss = loss + gate_entropy_weight * gate_entropy_loss(out)
                route_weight = float(cfg["loss"].get("lambda_route_supervision", 1.0))
                route_warmup = int(cfg["loss"].get("route_supervision_warmup", 0))
                if route_weight > 0 and epoch > route_warmup:
                    loss = loss + route_weight * route_supervision_loss(
                        out,
                        y,
                        tr,
                        temperature=float(cfg["loss"].get("route_supervision_temperature", 1.0)),
                    )
                route_entropy_weight = float(cfg["loss"].get("lambda_route_entropy", 0.0))
                if route_entropy_weight:
                    loss = loss + route_entropy_weight * route_entropy_loss(out)
                route_balance_weight = float(cfg["loss"].get("lambda_route_balance", 0.0))
                route_balance_warmup = int(cfg["loss"].get("route_balance_warmup", 0))
                if route_balance_weight > 0 and epoch > route_balance_warmup:
                    loss = loss + route_balance_weight * route_balance_loss(
                        out,
                        target=cfg["loss"].get("route_balance_target"),
                        temperature=float(cfg["loss"].get("route_balance_temperature", 1.0)),
                    )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("grad_clip", 5.0))
        opt.step()
        model.eval()
        with torch.no_grad():
            out = model(x, view_ops, gcn_adj, risk_features)
            val = masked_score(out["logits"], y, va, metric)
            test = masked_score(out["logits"], y, te, metric)
        select_score = _selection_score(cfg, val, test)
        if select_score > best_select:
            best_select = select_score
            best_val = val
            best_test = test
            best_epoch = epoch
            best = copy.deepcopy(model.state_dict())
            gate = out.get("gate")
            route = out.get("route_probs")
            if gate is not None:
                gate_safe = gate.clamp_min(1e-8)
                best_gate_mean = [float(v) for v in gate.mean(dim=0).detach().cpu().tolist()]
                best_gate_entropy = float((-(gate_safe * gate_safe.log()).sum(dim=1).mean()).detach().cpu().item())
            if route is not None:
                route_safe = route.clamp_min(1e-8)
                best_route_mean = [float(v) for v in route.mean(dim=0).detach().cpu().tolist()]
                best_route_entropy = float((-(route_safe * route_safe.log()).sum(dim=1).mean()).detach().cpu().item())
            fusion_mode = out.get("fusion_mode_probs")
            if fusion_mode is not None:
                best_fusion_mode_mean = [float(v) for v in fusion_mode.mean(dim=0).detach().cpu().tolist()]
            calibration_alpha = out.get("calibration_alpha")
            if calibration_alpha is not None:
                best_calibration_alpha_mean = [float(v) for v in calibration_alpha.mean(dim=0).detach().cpu().tolist()]
            best_path_val_scores = {
                name: masked_score(logits, y, va, metric)
                for name, logits in zip(path_names, out.get("path_logits_list", []))
            }
            best_path_test_scores = {
                name: masked_score(logits, y, te, metric)
                for name, logits in zip(path_names, out.get("path_logits_list", []))
            }
            bad = 0
        else:
            bad += 1
        if bad >= cfg["train"]["patience"]:
            break
    if best is not None:
        model.load_state_dict(best)
    best_routing_diagnostics = None
    if bool(cfg["train"].get("save_routing_diagnostics", True)):
        model.eval()
        with torch.no_grad():
            best_out = model(x, view_ops, gcn_adj, risk_features)
        best_routing_diagnostics = routing_diagnostics(
            best_out,
            y,
            tr,
            va,
            te,
            graph.adj,
            selected_views,
            path_names,
        )
    result = {
        "dataset": graph.name,
        "seed": seed,
        "split_id": graph.split_id,
        "variant": variant,
        "metric": metric,
        "views": selected_views,
        "regimes": path_names,
        "best_epoch": best_epoch,
        "best_val_score": best_val,
        "best_test_score": best_test,
        "best_select_score": best_select,
        "seconds": round(time.time() - start, 2),
    }
    if best_gate_mean is not None:
        result["best_gate_mean"] = best_gate_mean
        result["best_gate_entropy"] = best_gate_entropy
    if best_route_mean is not None:
        result["best_regime_mean"] = best_route_mean
        result["best_route_entropy"] = best_route_entropy
    if best_fusion_mode_mean is not None:
        result["fusion_mode_names"] = ["top1", "top2", "soft"]
        result["best_fusion_mode_mean"] = best_fusion_mode_mean
    if best_calibration_alpha_mean is not None:
        result["best_calibration_alpha_mean"] = dict(zip(path_names, best_calibration_alpha_mean))
    if best_path_val_scores is not None:
        result["best_path_val_scores"] = best_path_val_scores
        result["best_path_test_scores"] = best_path_test_scores
    if best_routing_diagnostics is not None:
        result["routing_diagnostics"] = best_routing_diagnostics
    if metric == "accuracy":
        result["best_val_acc"] = best_val
        result["best_test_acc"] = best_test
    return result


def train_once(cfg, seed, variant="baseline"):
    model_type = str(cfg["model"].get("model_type", "")).lower()
    if model_type in ("risk_router", "risk_router_hgnn", "r3_hgnn", "isolated_router"):
        return train_risk_router_once(cfg, seed, variant)
    if model_type in ("deep_gcn", "gcn_plus", "res_gcn") or variant in ("deep_gcn", "gcn_plus"):
        return train_deep_gnn_once(cfg, seed, variant)
    if model_type in ("mlp", "lp_mlp", "compat_mlp", "labelprop_mlp") or variant == "mlp":
        return train_mlp_once(cfg, seed, variant)
    flags = parse_variant(variant)
    set_seed(seed)
    graph = load_node_dataset(cfg["dataset"], seed)
    metric = str(cfg["dataset"].get("metric", "accuracy")).lower()
    enabled_views = cfg["model"].get("enabled_views", ["structural", "feature_knn", "prototype"])
    include_elastic = variant not in NO_AUX_VARIANTS
    include_knn_pairs = include_elastic and ("feature_knn" in enabled_views)
    builder = MultiViewHypergraphBuilder(
        hop_k=cfg["model"]["hop_k"],
        structural_cap=cfg["model"]["structural_cap"],
        knn_k=cfg["model"]["knn_k"],
        n_clusters=cfg["model"]["n_clusters"],
        elastic_margin=cfg["loss"]["elastic_margin"],
        max_elastic_pairs=cfg["loss"]["max_elastic_pairs"],
        random_state=seed,
        soft_membership=("soft_hg" in flags),
        prototype_method=cfg["model"].get("prototype_method", "kmeans"),
        prototype_soft_topk=cfg["model"].get("prototype_soft_topk", 2),
        structure_split_quantile=cfg["model"].get("structure_split_quantile", 0.5),
    )
    views = builder.build(
        graph.x.numpy(),
        graph.adj,
        enabled_views=enabled_views,
        include_elastic=include_elastic,
        include_knn_pairs=include_knn_pairs,
    )
    view_ops, selected_views, gate_prior = build_view_ops(views, cfg["model"])
    direct_experts = [str(name) for name in cfg["model"].get("direct_experts", [])]
    gate_prior_cfg = cfg["model"].get("view_gate_prior", {})
    for name in direct_experts:
        selected_views.append(name)
        gate_prior.append(float(gate_prior_cfg.get(name, 1.0)))
    regime_groups, regime_names, regime_prior = build_regime_groups(cfg["model"], selected_views)
    gcn_adj = None
    if any(name.lower() in ("gcn", "raw_gcn", "graph") for name in direct_experts):
        adj = graph.adj
        if bool(cfg["model"].get("raw_gcn_use_reverse_edges", False)):
            adj = (adj + adj.T).tocsr()
            adj.data = np.ones_like(adj.data, dtype=np.float32)
            adj.eliminate_zeros()
        gcn_adj = scipy_to_torch_sparse(
            normalize_adj(
                adj,
                self_loop=bool(cfg["model"].get("raw_gcn_self_loop", True)),
                mode=cfg["model"].get("raw_gcn_adj_norm", "row"),
            ).tocsr()
        )
    pair_index = torch.from_numpy(views.elastic_pairs).long()
    lower = torch.from_numpy(views.elastic_lower).float()
    upper = torch.from_numpy(views.elastic_upper).float()
    model = MVHGCL(
        graph.num_features,
        cfg["model"]["hidden_dim"],
        cfg["model"]["proj_dim"],
        graph.num_classes,
        num_views=len(view_ops),
        K=cfg["model"]["K"],
        dropout=cfg["model"]["dropout"],
        gate_temperature=cfg["model"].get("gate_temperature", 2.0),
        gate_residual=cfg["model"].get("gate_residual", 0.25),
        refine=cfg["model"].get("refine", False),
        gate_mode=cfg["model"].get("gate_mode", "soft"),
        reliability_entropy_bias=cfg["model"].get("reliability_entropy_bias", 0.0),
        fusion_mode=cfg["model"].get("fusion_mode", "embedding"),
        gate_detach_inputs=cfg["model"].get("gate_detach_inputs", False),
        gate_topk=cfg["model"].get("gate_topk", 0),
        gate_sharpen_power=cfg["model"].get("gate_sharpen_power", 1.0),
        gate_adaptive_sparse=cfg["model"].get("gate_adaptive_sparse", False),
        gate_adaptive_margin=cfg["model"].get("gate_adaptive_margin", 0.15),
        gate_adaptive_topk=cfg["model"].get("gate_adaptive_topk", 1),
        direct_experts=direct_experts,
        gcn_layers=cfg["model"].get("raw_gcn_layers", 2),
        gcn_residual=cfg["model"].get("raw_gcn_residual", True),
        gcn_norm=cfg["model"].get("raw_gcn_norm", "layer"),
        regime_groups=regime_groups,
        regime_prior=regime_prior,
        regime_temperature=cfg["model"].get("regime_temperature", 1.0),
        regime_topk=cfg["model"].get("regime_topk", 0),
        regime_sharpen_power=cfg["model"].get("regime_sharpen_power", 1.0),
        gate_prior=gate_prior,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    x, y, tr, va, te = graph.x, graph.y, graph.train_mask, graph.val_mask, graph.test_mask
    best = None
    best_val = -1.0
    best_test = -1.0
    best_select = -1.0
    best_epoch = 0
    best_gate_mean = None
    best_gate_entropy = None
    best_regime_mean = None
    bad = 0
    start = time.time()
    expert_pretrain_epochs = int(cfg["train"].get("expert_pretrain_epochs", 0))
    freeze_after_pretrain = bool(cfg["train"].get("freeze_experts_after_pretrain", False))
    label_smoothing = float(cfg.get("loss", {}).get("label_smoothing", 0.0))
    experts_frozen = False
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        if (
            freeze_after_pretrain
            and expert_pretrain_epochs > 0
            and epoch == expert_pretrain_epochs + 1
            and not experts_frozen
        ):
            freeze_expert_modules(model)
            trainable = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable, lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
            experts_frozen = True
        model.train()
        out = model(x, view_ops, gcn_adj)
        if epoch <= expert_pretrain_epochs:
            loss = view_ce_loss(out, y, tr, label_smoothing=label_smoothing)
        else:
            cls = F.cross_entropy(out["logits"][tr], y[tr], label_smoothing=label_smoothing)
            view_ce_weight = float(cfg["loss"].get("lambda_view_ce", 0.0))
            if view_ce_weight > 0 and out.get("view_logits_list") and not experts_frozen:
                cls = cls + view_ce_weight * view_ce_loss(out, y, tr, label_smoothing=label_smoothing)
            gate_sup_weight = float(cfg["loss"].get("lambda_gate_supervision", 0.0))
            gate_sup_warmup = int(cfg["loss"].get("gate_supervision_warmup", 0))
            if gate_sup_weight > 0 and epoch > gate_sup_warmup:
                gate_sup = gate_supervision_loss(
                    out,
                    y,
                    tr,
                    temperature=float(cfg["loss"].get("gate_supervision_temperature", 1.0)),
                )
                cls = cls + gate_sup_weight * gate_sup
            gate_entropy_weight = float(cfg["loss"].get("lambda_gate_entropy", 0.0))
            if gate_entropy_weight > 0:
                cls = cls + gate_entropy_weight * gate_entropy_loss(out)
            align, var, elastic = compute_aux_losses(cfg, flags, out, pair_index, lower, upper, y, tr, epoch, variant)
            loss = cls
            loss = loss + cfg["loss"]["lambda_align"] * align
            loss = loss + cfg["loss"]["lambda_var"] * var
            loss = loss + cfg["loss"]["lambda_elastic"] * elastic
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        opt.step()
        model.eval()
        with torch.no_grad():
            out = model(x, view_ops, gcn_adj)
            logits = out["logits"]
            val = masked_score(logits, y, va, metric)
            test = masked_score(logits, y, te, metric)
        select_score = _selection_score(cfg, val, test)
        if select_score > best_select:
            best_select = select_score
            best_val = val
            best_test = test
            best_epoch = epoch
            best = copy.deepcopy(model.state_dict())
            gate = out.get("gate")
            if gate is not None:
                gate_safe = gate.clamp_min(1e-8)
                best_gate_mean = [float(v) for v in gate.mean(dim=0).detach().cpu().tolist()]
                best_gate_entropy = float((-(gate_safe * gate_safe.log()).sum(dim=1).mean()).detach().cpu().item())
            regime_probs = out.get("regime_probs")
            if regime_probs is not None:
                best_regime_mean = [float(v) for v in regime_probs.mean(dim=0).detach().cpu().tolist()]
            bad = 0
        else:
            bad += 1
        if bad >= cfg["train"]["patience"]:
            break
    if best is not None:
        model.load_state_dict(best)
    best_routing_diagnostics = None
    if bool(cfg["train"].get("save_routing_diagnostics", True)):
        model.eval()
        with torch.no_grad():
            best_out = model(x, view_ops, gcn_adj)
        best_routing_diagnostics = routing_diagnostics(
            best_out,
            y,
            tr,
            va,
            te,
            graph.adj,
            selected_views,
            regime_names,
        )
    result = {
        "dataset": graph.name,
        "seed": seed,
        "split_id": graph.split_id,
        "variant": variant,
        "metric": metric,
        "views": selected_views,
        "regimes": regime_names,
        "best_epoch": best_epoch,
        "best_val_score": best_val,
        "best_test_score": best_test,
        "best_select_score": best_select,
        "seconds": round(time.time() - start, 2),
    }
    if best_gate_mean is not None:
        result["best_gate_mean"] = best_gate_mean
        result["best_gate_entropy"] = best_gate_entropy
    if best_regime_mean is not None:
        result["best_regime_mean"] = best_regime_mean
    if best_routing_diagnostics is not None:
        result["routing_diagnostics"] = best_routing_diagnostics
    if metric == "accuracy":
        result["best_val_acc"] = best_val
        result["best_test_acc"] = best_test
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--output-dir", default="results_node")
    ap.add_argument(
        "--selection",
        choices=["val", "oracle"],
        help="Override checkpoint selection. oracle uses max(val,test) and is only for exploratory upper-bound search.",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.selection:
        cfg.setdefault("train", {})["selection"] = args.selection
    if cfg.get("skip", False):
        reason = cfg.get("skip_reason", "configuration disabled")
        print(
            {
                "config": args.config,
                "variant": args.variant,
                "skipped": True,
                "reason": reason,
            }
        )
        return
    seeds = args.seeds or cfg.get("seeds", [cfg.get("seed", 42)])
    out_dir = os.path.join(args.output_dir, cfg["dataset"]["name"].lower(), args.variant)
    ensure_dir(out_dir)
    runs = []
    for seed in seeds:
        res = train_once(cfg, seed, args.variant)
        runs.append(res)
        save_json(res, os.path.join(out_dir, f"seed_{seed}.json"))
        print(res)
    best_run = max(runs, key=lambda r: r["best_test_score"])
    best_select_run = max(runs, key=lambda r: r.get("best_select_score", r["best_test_score"]))
    metric = runs[0]["metric"] if runs else str(cfg["dataset"].get("metric", "accuracy")).lower()
    summary = {
        "dataset": cfg["dataset"]["name"],
        "variant": args.variant,
        "metric": metric,
        "seeds": seeds,
        "runs": runs,
        "best_val_mean": mean([r["best_val_score"] for r in runs]),
        "best_test_mean": mean([r["best_test_score"] for r in runs]),
        "best_test_std": stdev([r["best_test_score"] for r in runs]) if len(runs) > 1 else 0.0,
        "best_single_test": best_run["best_test_score"],
        "best_single_seed": best_run["seed"],
        "best_single_split_id": best_run["split_id"],
        "best_single_epoch": best_run["best_epoch"],
        "best_select_score": best_select_run.get("best_select_score", best_select_run["best_test_score"]),
        "best_select_seed": best_select_run["seed"],
        "best_select_split_id": best_select_run["split_id"],
        "best_select_epoch": best_select_run["best_epoch"],
        "selection": str(cfg["train"].get("selection", "val")).lower(),
    }
    save_json(summary, os.path.join(out_dir, "summary.json"))
    print(summary)


if __name__ == "__main__":
    main()
