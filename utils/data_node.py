from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Hashable, List, TypeVar

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import scipy.sparse as sp
import torch
from scipy.io import loadmat


@dataclass
class GraphData:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    adj: sp.csr_matrix
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_classes: int
    num_features: int
    label_names: List[str]
    split_id: int


CITATION_FILES = {
    "cora": ("cora.content", "cora.cites"),
}

T = TypeVar("T")
_BASE_DATA_CACHE: Dict[tuple[Hashable, ...], object] = {}


def _cached_base_data(key: tuple[Hashable, ...], loader: Callable[[], T]) -> T:
    if key not in _BASE_DATA_CACHE:
        _BASE_DATA_CACHE[key] = loader()
    return _BASE_DATA_CACHE[key]  # type: ignore[return-value]


def _path_key(path: Path) -> str:
    return str(path.resolve())


def _normalize_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return (x / denom).astype(np.float32, copy=False)


def _finalize_adj(adj: sp.spmatrix, undirected: bool = False) -> sp.csr_matrix:
    adj = adj.tocsr().astype(np.float32)
    if undirected:
        adj = adj.maximum(adj.T).tocsr()
    adj = adj.tocoo()
    keep = adj.row != adj.col
    data = np.ones(int(keep.sum()), dtype=np.float32)
    adj = sp.coo_matrix((data, (adj.row[keep], adj.col[keep])), shape=adj.shape).tocsr()
    adj.sum_duplicates()
    if adj.nnz:
        adj.data = np.ones(adj.nnz, dtype=np.float32)
    return adj


def _mask_from_numpy(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(mask, dtype=np.bool_).copy()).bool()


def _resolve_split_idx(seed: int, num_splits: int) -> int:
    if num_splits <= 0:
        raise ValueError("num_splits must be positive")
    return int(seed) % int(num_splits)


def _stratified_counts(labels: np.ndarray, target_size: int) -> Dict[int, int]:
    classes, counts = np.unique(labels, return_counts=True)
    target_size = min(int(target_size), int(counts.sum()))
    if target_size <= 0:
        return {int(c): 0 for c in classes}
    raw = counts.astype(np.float64) * (target_size / float(counts.sum()))
    alloc = np.floor(raw).astype(np.int64)
    alloc = np.minimum(alloc, counts)
    remaining = target_size - int(alloc.sum())
    fractions = raw - alloc
    order = np.argsort(-fractions)
    while remaining > 0:
        changed = False
        for pos in order:
            if alloc[pos] < counts[pos]:
                alloc[pos] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break
    return {int(c): int(n) for c, n in zip(classes, alloc)}


def _stratified_val_test_split(
    remain: np.ndarray,
    y: np.ndarray,
    val_size: int,
    test_size: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    remain = np.asarray(remain, dtype=np.int64)
    labels = y[remain]
    val_counts = _stratified_counts(labels, int(val_size))
    val_idx, test_pool = [], []
    for c in np.unique(labels):
        idx = remain[labels == c].copy()
        rng.shuffle(idx)
        n_val = val_counts.get(int(c), 0)
        val_idx.extend(idx[:n_val].tolist())
        test_pool.extend(idx[n_val:].tolist())
    test_pool = np.asarray(test_pool, dtype=np.int64)
    test_labels = y[test_pool]
    test_counts = _stratified_counts(test_labels, int(test_size))
    test_idx = []
    for c in np.unique(test_labels):
        idx = test_pool[test_labels == c].copy()
        rng.shuffle(idx)
        n_test = test_counts.get(int(c), 0)
        test_idx.extend(idx[:n_test].tolist())
    return np.asarray(val_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def _stratified_node_sample(y: np.ndarray, max_nodes: int, seed: int) -> np.ndarray:
    valid = np.where(y >= 0)[0]
    if max_nodes <= 0 or valid.size <= max_nodes:
        return np.arange(y.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y[valid], return_counts=True)
    raw = counts.astype(np.float64) * (int(max_nodes) / float(counts.sum()))
    alloc = np.maximum(np.floor(raw).astype(np.int64), 1)
    alloc = np.minimum(alloc, counts)
    while int(alloc.sum()) > int(max_nodes):
        pos = int(np.argmax(alloc))
        if alloc[pos] <= 1:
            break
        alloc[pos] -= 1
    while int(alloc.sum()) < int(max_nodes):
        fractions = raw - alloc
        order = np.argsort(-fractions)
        changed = False
        for pos in order:
            if alloc[pos] < counts[pos]:
                alloc[pos] += 1
                changed = True
                break
        if not changed:
            break
    selected = []
    for c, n_take in zip(classes, alloc):
        idx = valid[y[valid] == c].copy()
        rng.shuffle(idx)
        selected.extend(idx[: int(n_take)].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _induce_node_subset(x: np.ndarray, y: np.ndarray, adj: sp.csr_matrix, idx: np.ndarray):
    idx = np.asarray(idx, dtype=np.int64)
    return x[idx], y[idx], adj.tocsr()[idx][:, idx].tocsr()


def _random_project_features(x: np.ndarray, dim: int, seed: int) -> np.ndarray:
    dim = int(dim)
    if dim <= 0 or x.shape[1] <= dim:
        return _normalize_features(x)
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0 / np.sqrt(float(x.shape[1])), size=(x.shape[1], dim)).astype(np.float32)
    return _normalize_features(x @ w)


def _build_masks(y: np.ndarray, seed: int, train_per_class: int, val_size: int, test_size: int):
    rng = np.random.default_rng(seed)
    train_idx = []
    valid_y = y[y >= 0]
    for c in np.unique(valid_y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        train_idx.extend(idx[: min(train_per_class, len(idx))].tolist())
    train_idx = np.array(sorted(set(train_idx)), dtype=np.int64)
    valid_idx = np.where(y >= 0)[0]
    remain = np.setdiff1d(valid_idx, train_idx)
    train_size = val_size
    eval_size = test_size
    if isinstance(val_size, int) and isinstance(test_size, int) and (val_size + test_size) > len(remain):
        if len(remain) < 2:
            raise ValueError("Not enough labeled nodes left after train split to build val/test masks")
        ratio = float(val_size) / float(val_size + test_size)
        train_size = max(1, int(round(len(remain) * ratio)))
        train_size = min(train_size, len(remain) - 1)
        eval_size = len(remain) - train_size
    val_idx, test_idx = _stratified_val_test_split(remain, y, train_size, eval_size, seed)
    out = []
    for idx in [train_idx, val_idx, test_idx]:
        mask = torch.zeros(len(y), dtype=torch.bool)
        mask[torch.as_tensor(idx)] = True
        out.append(mask)
    return out, -1


def _read_content(path: Path):
    node_ids, feats, labels = [], [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            node_ids.append(parts[0])
            feats.append([float(v) for v in parts[1:-1]])
            labels.append(parts[-1])
    label_names = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(label_names)}
    x = _normalize_features(np.asarray(feats, dtype=np.float32))
    y = np.asarray([label_to_id[label] for label in labels], dtype=np.int64)
    id_map = {node_id: i for i, node_id in enumerate(node_ids)}
    return x, y, id_map, label_names


def _read_cites(path: Path, id_map: Dict[str, int], n: int):
    rows, cols = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            cited, citing = parts
            if cited not in id_map or citing not in id_map:
                continue
            a = id_map[citing]
            b = id_map[cited]
            rows.extend([a, b])
            cols.extend([b, a])
    adj = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()
    return _finalize_adj(adj, undirected=True)


def _read_citation_base(data_dir: Path, content_name: str, cites_name: str):
    x, y, id_map, label_names = _read_content(data_dir / content_name)
    adj = _read_cites(data_dir / cites_name, id_map, len(y))
    return x, y, adj, label_names


def _parse_geom_gcn_node_file(path: Path):
    rows = []
    max_feat_idx = -1
    dense_mode = None
    with path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()[1:]
    for line in lines:
        if not line:
            continue
        node_id, feat_field, label = line.split("\t")
        tokens = [tok for tok in feat_field.split(",") if tok != ""]
        if dense_mode is None:
            dense_mode = any("." in tok for tok in tokens) or len(tokens) > 64
        if dense_mode:
            feat = np.asarray([float(v) for v in tokens], dtype=np.float32)
            rows.append((int(node_id), feat, int(label)))
        else:
            idx = [int(v) for v in tokens]
            max_feat_idx = max(max_feat_idx, max(idx) if idx else -1)
            rows.append((int(node_id), idx, int(label)))
    num_nodes = max(node_id for node_id, _, _ in rows) + 1 if rows else 0
    if dense_mode:
        feat_dim = rows[0][1].shape[0] if rows else 0
        x = np.zeros((num_nodes, feat_dim), dtype=np.float32)
        y = np.zeros((num_nodes,), dtype=np.int64)
        for node_id, feat, label in rows:
            x[node_id] = feat
            y[node_id] = label
    else:
        feat_dim = max_feat_idx + 1
        x = np.zeros((num_nodes, feat_dim), dtype=np.float32)
        y = np.zeros((num_nodes,), dtype=np.int64)
        for node_id, feat_idx, label in rows:
            if feat_idx:
                x[node_id, feat_idx] = 1.0
            y[node_id] = label
    x = _normalize_features(x)
    label_names = [str(i) for i in sorted(set(y.tolist()))]
    return x, y, label_names


def _read_geom_gcn_edges(path: Path, n: int):
    rows, cols = [], []
    with path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()[1:]
    for line in lines:
        if not line:
            continue
        s, t = [int(v) for v in line.split("\t")]
        rows.extend([s, t])
        cols.extend([t, s])
    adj = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()
    return _finalize_adj(adj, undirected=True)


def _read_geom_gcn_base(data_dir: Path):
    x, y, label_names = _parse_geom_gcn_node_file(data_dir / "out1_node_feature_label.txt")
    adj = _read_geom_gcn_edges(data_dir / "out1_graph_edges.txt", len(y))
    return x, y, adj, label_names


def _load_geom_gcn_split(data_dir: Path, name: str, seed: int):
    split_files = sorted(data_dir.glob(f"{name}_split_0.6_0.2_*.npz"))
    if not split_files:
        return None
    split_idx = _resolve_split_idx(seed, len(split_files))
    raw = np.load(split_files[split_idx], allow_pickle=False)
    train_mask = _mask_from_numpy(raw["train_mask"])
    val_mask = _mask_from_numpy(raw["val_mask"])
    test_mask = _mask_from_numpy(raw["test_mask"])
    return train_mask, val_mask, test_mask, split_idx


def _read_heterophily_npz_base(path: Path, to_undirected: bool = True):
    raw = np.load(path, allow_pickle=False)
    x = raw["node_features"].astype(np.float32)
    x = _normalize_features(x)
    y = raw["node_labels"].astype(np.int64)
    edges = raw["edges"].astype(np.int64)
    if to_undirected:
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
    else:
        rows = edges[:, 0]
        cols = edges[:, 1]
    adj = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(x.shape[0], x.shape[0])).tocsr()
    train_masks = raw["train_masks"]
    val_masks = raw["val_masks"]
    test_masks = raw["test_masks"]
    label_names = [str(i) for i in sorted(set(y.tolist()))]
    return x, y, _finalize_adj(adj, undirected=to_undirected), label_names, train_masks, val_masks, test_masks


def _read_simple_npz(path: Path):
    raw = np.load(path, allow_pickle=False)
    x = raw["features"].astype(np.float32)
    x = _normalize_features(x)
    y = raw["target"].astype(np.int64)
    edges = raw["edges"].astype(np.int64)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    adj = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(x.shape[0], x.shape[0])).tocsr()
    label_names = [str(i) for i in sorted(set(y.tolist()))]
    return x, y, _finalize_adj(adj, undirected=True), label_names


def _read_linkx_mat(path: Path):
    mat = loadmat(path)
    A = mat["A"].tocsr().tocoo()
    rows = np.concatenate([A.row, A.col])
    cols = np.concatenate([A.col, A.row])
    n = A.shape[0]
    adj = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()

    metadata = mat["local_info"].astype(np.int64)
    y = metadata[:, 1].astype(np.int64) - 1
    y[y > 1] = -1
    x_fields = np.concatenate([metadata[:, :1], metadata[:, 2:]], axis=1)
    xs = []
    for i in range(x_fields.shape[1]):
        col = x_fields[:, i]
        uniq, inv = np.unique(col, return_inverse=True)
        one_hot = np.zeros((n, uniq.shape[0]), dtype=np.float32)
        one_hot[np.arange(n), inv] = 1.0
        xs.append(one_hot)
    x = np.concatenate(xs, axis=1).astype(np.float32)
    x = _normalize_features(x)
    label_names = [str(i) for i in sorted(set(y[y >= 0].tolist()))]
    return x, y.astype(np.int64), _finalize_adj(adj, undirected=True), label_names


def _read_pyg_coauthor_base(data_dir: Path, pyg_name: str):
    try:
        from torch_geometric.datasets import Coauthor
    except Exception as exc:
        raise RuntimeError(
            "Coauthor-CS/Physics requires torch_geometric. Install torch_geometric or use datasets already cached in configs/node."
        ) from exc
    dataset = Coauthor(root=str(data_dir), name=str(pyg_name))
    data = dataset[0]
    x = data.x.detach().cpu().numpy().astype(np.float32)
    x = _normalize_features(x)
    y = data.y.detach().cpu().numpy().astype(np.int64)
    edge_index = data.edge_index.detach().cpu().numpy().astype(np.int64)
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1], dtype=np.float32), (edge_index[0], edge_index[1])),
        shape=(x.shape[0], x.shape[0]),
    ).tocsr()
    label_names = [str(i) for i in sorted(set(y.tolist()))]
    return x, y, _finalize_adj(adj, undirected=True), label_names


def _load_linkx_split(path: Path, num_nodes: int, seed: int):
    raw = np.load(path, allow_pickle=True)
    split_idx = _resolve_split_idx(seed, len(raw))
    split = raw[split_idx].item() if hasattr(raw[split_idx], "item") else raw[split_idx]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[torch.as_tensor(split["train"], dtype=torch.long)] = True
    val_mask[torch.as_tensor(split["valid"], dtype=torch.long)] = True
    test_mask[torch.as_tensor(split["test"], dtype=torch.long)] = True
    return train_mask, val_mask, test_mask, split_idx


def load_node_dataset(cfg_dataset: Dict, seed: int):
    dtype = cfg_dataset.get("type", "citation").lower()
    name = cfg_dataset["name"]
    data_dir = Path(cfg_dataset["data_dir"])
    if dtype == "citation":
        content_name, cites_name = CITATION_FILES[name.lower()]
        x, y, adj, label_names = _cached_base_data(
            ("citation", name.lower(), _path_key(data_dir), content_name, cites_name),
            lambda: _read_citation_base(data_dir, content_name, cites_name),
        )
        masks, split_id = _build_masks(
            y,
            seed,
            cfg_dataset["train_per_class"],
            cfg_dataset["val_size"],
            cfg_dataset["test_size"],
        )
        train_mask, val_mask, test_mask = masks
    elif dtype in ("webkb", "actor", "geom_gcn", "wikipedia"):
        x, y, adj, label_names = _cached_base_data(
            ("geom_gcn", name.lower(), _path_key(data_dir)),
            lambda: _read_geom_gcn_base(data_dir),
        )
        split = _load_geom_gcn_split(data_dir, name, seed) if cfg_dataset.get("use_official_splits", True) else None
        if split is not None:
            train_mask, val_mask, test_mask, split_id = split
        else:
            masks, split_id = _build_masks(
                y,
                seed,
                cfg_dataset.get("train_per_class", 8),
                cfg_dataset.get("val_size", 36),
                cfg_dataset.get("test_size", 107),
            )
            train_mask, val_mask, test_mask = masks
    elif dtype == "heterophily_npz":
        npz_path = data_dir / cfg_dataset.get("file_name", f"{name}.npz")
        to_undirected = bool(cfg_dataset.get("to_undirected", True))
        x, y, adj, label_names, train_masks, val_masks, test_masks = _cached_base_data(
            ("heterophily_npz", name.lower(), _path_key(npz_path), to_undirected),
            lambda: _read_heterophily_npz_base(npz_path, to_undirected=to_undirected),
        )
        split_id = _resolve_split_idx(seed, train_masks.shape[0])
        train_mask = _mask_from_numpy(train_masks[split_id])
        val_mask = _mask_from_numpy(val_masks[split_id])
        test_mask = _mask_from_numpy(test_masks[split_id])
    elif dtype == "simple_npz":
        npz_path = data_dir / cfg_dataset.get("file_name", f"{name}.npz")
        x, y, adj, label_names = _cached_base_data(
            ("simple_npz", name.lower(), _path_key(npz_path)),
            lambda: _read_simple_npz(npz_path),
        )
        masks, split_id = _build_masks(
            y,
            seed,
            cfg_dataset.get("train_per_class", 20),
            cfg_dataset.get("val_size", 500),
            cfg_dataset.get("test_size", 1000),
        )
        train_mask, val_mask, test_mask = masks
    elif dtype == "linkx_mat":
        mat_path = data_dir / cfg_dataset.get("file_name", "data.mat")
        x, y, adj, label_names = _cached_base_data(
            ("linkx_mat", name.lower(), _path_key(mat_path)),
            lambda: _read_linkx_mat(mat_path),
        )
        split_file = cfg_dataset.get("split_file")
        labeled_mask = torch.from_numpy((y >= 0)).bool()
        if split_file:
            train_mask, val_mask, test_mask, split_id = _load_linkx_split(data_dir / split_file, len(y), seed)
            train_mask &= labeled_mask
            val_mask &= labeled_mask
            test_mask &= labeled_mask
        else:
            masks, split_id = _build_masks(
                y,
                seed,
                cfg_dataset.get("train_per_class", 20),
                cfg_dataset.get("val_size", 500),
                cfg_dataset.get("test_size", 1000),
            )
            train_mask, val_mask, test_mask = masks
    elif dtype == "pyg_coauthor":
        pyg_name = cfg_dataset.get("pyg_name", "CS")
        x, y, adj, label_names = _cached_base_data(
            ("pyg_coauthor", str(pyg_name).lower(), _path_key(data_dir)),
            lambda: _read_pyg_coauthor_base(data_dir, pyg_name),
        )
        masks, split_id = _build_masks(
            y,
            seed,
            cfg_dataset.get("train_per_class", 20),
            cfg_dataset.get("val_size", 1000),
            cfg_dataset.get("test_size", 1000),
        )
        train_mask, val_mask, test_mask = masks
    else:
        raise ValueError(f"Unsupported node dataset type: {dtype}")
    max_nodes = int(cfg_dataset.get("max_nodes", 0) or 0)
    feature_projection_dim = int(cfg_dataset.get("feature_projection_dim", 0) or 0)
    if max_nodes > 0 and len(y) > max_nodes:
        sample_idx = _stratified_node_sample(y, max_nodes=max_nodes, seed=seed)
        x, y, adj = _induce_node_subset(x, y, adj, sample_idx)
        masks, split_id = _build_masks(
            y,
            seed,
            cfg_dataset.get("train_per_class", 20),
            cfg_dataset.get("val_size", max(1, int(0.2 * len(y)))),
            cfg_dataset.get("test_size", max(1, int(0.2 * len(y)))),
        )
        train_mask, val_mask, test_mask = masks
    if feature_projection_dim > 0:
        x = _random_project_features(x, feature_projection_dim, seed + int(cfg_dataset.get("projection_seed_offset", 7919)))
    return GraphData(
        name=name,
        x=torch.from_numpy(np.asarray(x, dtype=np.float32).copy()),
        y=torch.from_numpy(np.asarray(y, dtype=np.int64).copy()),
        adj=adj,
        train_mask=train_mask.detach().clone().bool(),
        val_mask=val_mask.detach().clone().bool(),
        test_mask=test_mask.detach().clone().bool(),
        num_classes=len(label_names),
        num_features=x.shape[1],
        label_names=label_names,
        split_id=split_id,
    )
