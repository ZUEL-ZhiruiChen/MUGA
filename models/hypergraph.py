from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


@dataclass
class HypergraphViews:
    incidences: List[sp.csr_matrix]
    node_degree_inv: List[np.ndarray]
    edge_degree_inv: List[np.ndarray]
    view_names: List[str]
    hyperedge_counts: List[int]
    elastic_pairs: np.ndarray
    elastic_lower: np.ndarray
    elastic_upper: np.ndarray


class MultiViewHypergraphBuilder:
    def __init__(
        self,
        hop_k=2,
        structural_cap=20,
        knn_k=10,
        n_clusters=64,
        elastic_margin=0.18,
        max_elastic_pairs=10000,
        random_state=42,
        soft_membership=False,
        prototype_method="kmeans",
        prototype_soft_topk=2,
        structure_split_quantile=0.5,
    ):
        self.hop_k = hop_k
        self.structural_cap = structural_cap
        self.knn_k = knn_k
        self.n_clusters = n_clusters
        self.elastic_margin = elastic_margin
        self.max_elastic_pairs = max_elastic_pairs
        self.random_state = random_state
        self.soft_membership = soft_membership
        self.prototype_method = prototype_method
        self.prototype_soft_topk = prototype_soft_topk
        self.structure_split_quantile = structure_split_quantile

    def _adj_list(self, adj):
        return [adj.indices[adj.indptr[i] : adj.indptr[i + 1]] for i in range(adj.shape[0])]

    def _structural_hyperedges(self, adj):
        neigh = self._adj_list(adj)
        hes = []
        for node in range(adj.shape[0]):
            visited = {node}
            frontier = [node]
            collected = [(node, 1.0)]
            for depth in range(1, self.hop_k + 1):
                nxt = []
                for cur in frontier:
                    for nb in neigh[cur]:
                        nb = int(nb)
                        if nb not in visited:
                            visited.add(nb)
                            nxt.append(nb)
                if not nxt:
                    break
                w = 1.0 / (depth + 1)
                collected.extend((nb, w) for nb in nxt)
                frontier = nxt
                if len(collected) >= self.structural_cap:
                    break
            hes.append(collected[: self.structural_cap])
        return hes

    def _identity_hyperedges(self, n):
        return [[(node, 1.0)] for node in range(int(n))]

    def _split_adj_by_feature_similarity(self, adj, x):
        coo = adj.tocoo()
        if coo.nnz == 0:
            empty = sp.csr_matrix(adj.shape, dtype=np.float32)
            return empty, empty
        sims = np.asarray((x[coo.row] * x[coo.col]).sum(axis=1)).reshape(-1)
        q = float(np.clip(self.structure_split_quantile, 0.05, 0.95))
        threshold = float(np.quantile(sims, q))
        homo_mask = sims >= threshold
        hetero_mask = ~homo_mask
        homo = sp.coo_matrix(
            (np.ones(int(homo_mask.sum()), dtype=np.float32), (coo.row[homo_mask], coo.col[homo_mask])),
            shape=adj.shape,
        ).tocsr()
        hetero = sp.coo_matrix(
            (np.ones(int(hetero_mask.sum()), dtype=np.float32), (coo.row[hetero_mask], coo.col[hetero_mask])),
            shape=adj.shape,
        ).tocsr()
        homo.setdiag(0)
        hetero.setdiag(0)
        homo.eliminate_zeros()
        hetero.eliminate_zeros()
        return homo, hetero

    def _knn_hyperedges(self, x):
        n = int(x.shape[0])
        if n <= 1:
            idx = np.zeros((n, 1), dtype=np.int64)
            return [[(0, 1.0)]] if n == 1 else [], idx
        n_neighbors = min(self.knn_k + 1, max(1, n - 1))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn.fit(x)
        dists, idx = nn.kneighbors(return_distance=True)
        hes = []
        for row_idx, row_dist in zip(idx, dists):
            nodes = []
            for v, d in zip(row_idx.tolist(), row_dist.tolist()):
                w = max(1e-3, 1.0 - d) if self.soft_membership else 1.0
                nodes.append((int(v), float(w)))
            hes.append(nodes)
        return hes, idx

    def _fit_prototypes(self, x, n_clusters):
        if n_clusters == 1:
            return np.zeros((x.shape[0],), dtype=np.int64), normalize(x.mean(axis=0, keepdims=True))
        if self.prototype_method == "kmeans":
            if x.shape[0] > 4096:
                estimator = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state,
                    batch_size=min(4096, x.shape[0]),
                    n_init=10,
                    max_iter=100,
                )
            else:
                estimator = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state,
                    n_init=10,
                    max_iter=100,
                )
            labels = estimator.fit_predict(x)
            centers = normalize(estimator.cluster_centers_.astype(np.float32))
            return labels.astype(np.int64), centers.astype(np.float32)
        rng = np.random.default_rng(self.random_state)
        anchor_idx = rng.choice(x.shape[0], size=n_clusters, replace=False)
        centers = normalize(x[anchor_idx].astype(np.float32))
        sims = x @ centers.T
        return sims.argmax(axis=1).astype(np.int64), centers.astype(np.float32)

    def _prototype_hyperedges(self, x):
        n = int(x.shape[0])
        if n <= 1:
            return [[(0, 1.0)]] if n == 1 else []
        n_clusters = max(1, min(int(self.n_clusters), n))
        labels, centers = self._fit_prototypes(x, n_clusters)
        if not self.soft_membership:
            hes = []
            for c in range(n_clusters):
                nodes = np.where(labels == c)[0].tolist()
                if nodes:
                    hes.append([(int(v), 1.0) for v in nodes])
            return hes
        topk = min(max(1, int(self.prototype_soft_topk)), n_clusters)
        if topk == 1:
            hes = [[] for _ in range(n_clusters)]
            for node_id, cluster_id in enumerate(labels.tolist()):
                hes[cluster_id].append((node_id, 1.0))
            return [he for he in hes if he]
        sims = x @ centers.T
        topk_ids = np.argpartition(-sims, kth=topk - 1, axis=1)[:, :topk]
        hes = [[] for _ in range(n_clusters)]
        for i in range(n):
            ids = topk_ids[i]
            vals = sims[i, ids]
            vals = vals - vals.max()
            probs = np.exp(vals)
            probs = probs / probs.sum()
            for c, prob in zip(ids.tolist(), probs.tolist()):
                hes[c].append((i, float(max(prob, 1e-3))))
        return [he for he in hes if he]

    def _incidence(self, n, hyperedges):
        rows, cols, data = [], [], []
        edge_id = 0
        for nodes in hyperedges:
            seen = {}
            for v, w in nodes:
                if 0 <= int(v) < n:
                    seen[int(v)] = max(float(w), seen.get(int(v), 0.0))
            if not seen:
                continue
            for v, w in sorted(seen.items()):
                rows.append(v)
                cols.append(edge_id)
                data.append(w)
            edge_id += 1
        return sp.coo_matrix(
            (np.asarray(data, dtype=np.float32), (rows, cols)),
            shape=(n, edge_id),
        ).tocsr()

    def _degree_stats(self, H):
        H = H.copy().astype(np.float32)
        H.data = np.clip(H.data, 1e-6, None)
        edge_degree = np.asarray(H.sum(axis=0)).reshape(-1).astype(np.float32)
        edge_degree[edge_degree == 0] = 1.0
        node_degree = np.asarray(H.sum(axis=1)).reshape(-1).astype(np.float32)
        node_degree[node_degree == 0] = 1.0
        return 1.0 / node_degree, 1.0 / edge_degree

    def _adj_pairs(self, adj):
        tri = sp.triu(adj, k=1).tocoo()
        return np.vstack([tri.row, tri.col]).astype(np.int64)

    def _mutual_knn_pairs(self, idx):
        sets = [set(r[1:].tolist()) for r in idx]
        pairs = set()
        for i, s in enumerate(sets):
            for j in s:
                if i in sets[j]:
                    a, b = sorted((i, int(j)))
                    if a != b:
                        pairs.add((a, b))
        if not pairs:
            return np.zeros((2, 0), dtype=np.int64)
        return np.array(sorted(pairs), dtype=np.int64).T

    def _sample_pairs(self, pairs, target, rng):
        if pairs.shape[1] <= target:
            return pairs
        idx = rng.choice(pairs.shape[1], size=target, replace=False)
        return pairs[:, idx]

    def _resolve_pair_budget(self, n, total_pairs):
        if total_pairs <= 0:
            return 0
        base_budget = self.max_elastic_pairs
        adaptive_floor = min(total_pairs, max(2048, min(int(n), 20000)))
        if base_budget is None or int(base_budget) <= 0:
            return adaptive_floor
        return min(total_pairs, max(int(base_budget), adaptive_floor))

    def _merge_unique_pairs(self, pair_sources):
        valid = [p.T for p in pair_sources if p.size > 0]
        if not valid:
            return np.zeros((2, 0), dtype=np.int64)
        merged = np.concatenate(valid, axis=0)
        merged = np.unique(merged, axis=0)
        return merged.T.astype(np.int64)

    def _elastic(self, adj, knn_idx, x):
        graph_pairs = self._adj_pairs(adj)
        knn_pairs = self._mutual_knn_pairs(knn_idx) if knn_idx is not None else np.zeros((2, 0), dtype=np.int64)
        total_pairs = graph_pairs.shape[1] + knn_pairs.shape[1]
        target = self._resolve_pair_budget(x.shape[0], total_pairs)
        if target <= 0:
            empty_pairs = np.zeros((2, 0), dtype=np.int64)
            empty_bounds = np.zeros((0,), dtype=np.float32)
            return empty_pairs, empty_bounds, empty_bounds
        rng = np.random.default_rng(self.random_state)
        half = max(1, target // 2)
        sampled_graph = self._sample_pairs(graph_pairs, min(graph_pairs.shape[1], half), rng)
        sampled_knn = self._sample_pairs(knn_pairs, min(knn_pairs.shape[1], target - sampled_graph.shape[1]), rng)
        pairs = self._merge_unique_pairs([sampled_graph, sampled_knn])
        if pairs.shape[1] < target:
            remainder = self._merge_unique_pairs([graph_pairs, knn_pairs])
            remainder = self._sample_pairs(remainder, min(remainder.shape[1], target), rng)
            pairs = remainder
        if pairs.shape[1] == 0:
            empty_bounds = np.zeros((0,), dtype=np.float32)
            return pairs, empty_bounds, empty_bounds
        src, dst = pairs[0], pairs[1]
        base = np.clip(1.0 - (x[src] * x[dst]).sum(axis=1), 0.0, 2.0)
        lower = np.clip(base - self.elastic_margin, 0.0, 2.0).astype(np.float32)
        upper = np.clip(base + self.elastic_margin, 0.0, 2.0).astype(np.float32)
        return pairs, lower, upper

    def build(self, x, adj, enabled_views=None, include_elastic=True, include_knn_pairs=True):
        enabled = set(enabled_views or ["structural", "feature_knn", "prototype"])
        defs = []
        knn_idx = None
        if "identity" in enabled or "self" in enabled:
            defs.append(("identity", self._identity_hyperedges(x.shape[0])))
        if "structural" in enabled:
            defs.append(("structural", self._structural_hyperedges(adj)))
        split_requested = bool(
            {"structure_homo", "structure_hetero", "structural_homo", "structural_hetero"} & enabled
        )
        if split_requested:
            homo_adj, hetero_adj = self._split_adj_by_feature_similarity(adj, x)
            if "structure_homo" in enabled or "structural_homo" in enabled:
                defs.append(("structure_homo", self._structural_hyperedges(homo_adj)))
            if "structure_hetero" in enabled or "structural_hetero" in enabled:
                defs.append(("structure_hetero", self._structural_hyperedges(hetero_adj)))
        if "feature_knn" in enabled or include_knn_pairs:
            knn_hes, knn_idx = self._knn_hyperedges(x)
            if "feature_knn" in enabled:
                defs.append(("feature_knn", knn_hes))
        if "prototype" in enabled:
            defs.append(("prototype", self._prototype_hyperedges(x)))
        incidences, node_inv, edge_inv, counts, names = [], [], [], [], []
        for name, hes in defs:
            H = self._incidence(x.shape[0], hes)
            incidences.append(H)
            dv_inv, de_inv = self._degree_stats(H)
            node_inv.append(dv_inv)
            edge_inv.append(de_inv)
            counts.append(H.shape[1])
            names.append(name)
        if include_elastic:
            pairs, lower, upper = self._elastic(adj, knn_idx, x)
        else:
            pairs = np.zeros((2, 0), dtype=np.int64)
            lower = np.zeros((0,), dtype=np.float32)
            upper = np.zeros((0,), dtype=np.float32)
        return HypergraphViews(incidences, node_inv, edge_inv, names, counts, pairs, lower, upper)
