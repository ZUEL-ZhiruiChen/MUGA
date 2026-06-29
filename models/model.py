from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

torch.set_num_threads(1)


def hypergraph_propagate(op, x):
    # Apply Dv^-1 H De^-1 H^T to dense node features without materializing an NxN transition.
    msg = torch.sparse.mm(op["ht"], x)
    msg = msg * op["edge_degree_inv"].unsqueeze(-1)
    out = torch.sparse.mm(op["h"], msg)
    return out * op["node_degree_inv"].unsqueeze(-1)


class MLP(nn.Module):
    def __init__(self, dims, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.PReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FeatureMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=3, dropout=0.3, layer_norm=True):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * max(1, int(num_layers) - 1) + [num_classes]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.PReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepGCN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_classes,
        num_layers=4,
        dropout=0.5,
        residual=True,
        norm="batch",
    ):
        super().__init__()
        self.dropout = dropout
        self.residual = residual
        self.input = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(max(1, int(num_layers)))])
        norms = []
        for _ in self.layers:
            if str(norm).lower() == "batch":
                norms.append(nn.BatchNorm1d(hidden_dim))
            elif str(norm).lower() == "layer":
                norms.append(nn.LayerNorm(hidden_dim))
            else:
                norms.append(nn.Identity())
        self.norms = nn.ModuleList(norms)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj):
        h = self.input(x)
        h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
        for lin, norm in zip(self.layers, self.norms):
            msg = torch.sparse.mm(adj, h)
            upd = lin(msg)
            upd = norm(upd)
            upd = F.relu(upd)
            upd = F.dropout(upd, p=self.dropout, training=self.training)
            h = h + upd if self.residual else upd
        return self.out(h)


class ViewEncoder(nn.Module):
    def __init__(self, hidden_dim, proj_dim, K, dropout, refine=False):
        super().__init__()
        self.K = K
        self.pre = MLP([hidden_dim, hidden_dim, hidden_dim], dropout) if refine else None
        self.refine_block = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if refine
            else None
        )
        self.theta_logits = nn.Parameter(torch.zeros(K + 1))
        self.projector = MLP([hidden_dim, hidden_dim, proj_dim], dropout)
        self.predictor = MLP([proj_dim, proj_dim, proj_dim], dropout)

    def forward(self, op, h):
        cur = self.pre(h) if self.pre is not None else h
        theta = F.softmax(self.theta_logits, dim=0)
        out = theta[0] * cur
        prop = cur
        for k in range(1, self.K + 1):
            prop = hypergraph_propagate(op, prop)
            out = out + theta[k] * prop
        if self.refine_block is not None:
            out = out + self.refine_block(out)
        z = self.projector(out)
        p = self.predictor(z)
        return z, p, theta


class FeatureExpert(nn.Module):
    def __init__(self, hidden_dim, proj_dim, dropout):
        super().__init__()
        self.projector = MLP([hidden_dim, hidden_dim, proj_dim], dropout)
        self.predictor = MLP([proj_dim, proj_dim, proj_dim], dropout)

    def forward(self, h):
        z = self.projector(h)
        p = self.predictor(z)
        return z, p


class RawGCNExpert(nn.Module):
    def __init__(self, hidden_dim, proj_dim, num_layers=2, dropout=0.3, residual=True, norm="layer"):
        super().__init__()
        self.dropout = dropout
        self.residual = residual
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(max(1, int(num_layers)))])
        norms = []
        for _ in self.layers:
            if str(norm).lower() == "batch":
                norms.append(nn.BatchNorm1d(hidden_dim))
            elif str(norm).lower() == "layer":
                norms.append(nn.LayerNorm(hidden_dim))
            else:
                norms.append(nn.Identity())
        self.norms = nn.ModuleList(norms)
        self.projector = MLP([hidden_dim, hidden_dim, proj_dim], dropout)
        self.predictor = MLP([proj_dim, proj_dim, proj_dim], dropout)

    def forward(self, h, adj):
        if adj is None:
            raise ValueError("raw GCN expert requires normalized adjacency")
        cur = h
        for lin, norm in zip(self.layers, self.norms):
            msg = torch.sparse.mm(adj, cur)
            upd = lin(msg)
            upd = norm(upd)
            upd = F.relu(upd)
            upd = F.dropout(upd, p=self.dropout, training=self.training)
            cur = cur + upd if self.residual else upd
        z = self.projector(cur)
        p = self.predictor(z)
        return z, p


class MVHGCL(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        proj_dim,
        num_classes,
        num_views=3,
        K=3,
        dropout=0.3,
        gate_temperature=2.0,
        gate_residual=0.25,
        refine=False,
        gate_mode="soft",
        gate_prior=None,
        reliability_entropy_bias=0.0,
        fusion_mode="embedding",
        gate_detach_inputs=False,
        gate_topk=0,
        gate_sharpen_power=1.0,
        gate_adaptive_sparse=False,
        gate_adaptive_margin=0.15,
        gate_adaptive_topk=1,
        direct_experts=None,
        gcn_layers=2,
        gcn_residual=True,
        gcn_norm="layer",
        regime_groups=None,
        regime_prior=None,
        regime_temperature=1.0,
        regime_topk=0,
        regime_sharpen_power=1.0,
    ):
        super().__init__()
        self.hypergraph_num_views = num_views
        self.direct_expert_names = [str(name) for name in (direct_experts or [])]
        self.num_views = num_views + len(self.direct_expert_names)
        self.gate_temperature = gate_temperature
        self.gate_residual = gate_residual
        self.gate_mode = gate_mode
        self.reliability_entropy_bias = reliability_entropy_bias
        self.fusion_mode = fusion_mode
        self.gate_detach_inputs = gate_detach_inputs
        self.gate_topk = int(gate_topk or 0)
        self.gate_sharpen_power = float(gate_sharpen_power)
        self.gate_adaptive_sparse = bool(gate_adaptive_sparse)
        self.gate_adaptive_margin = float(gate_adaptive_margin)
        self.gate_adaptive_topk = max(1, int(gate_adaptive_topk or 1))
        self.regime_temperature = float(regime_temperature)
        self.regime_topk = int(regime_topk or 0)
        self.regime_sharpen_power = float(regime_sharpen_power)
        self.input_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.views = nn.ModuleList([ViewEncoder(hidden_dim, proj_dim, K, dropout, refine=refine) for _ in range(num_views)])
        direct_modules = {}
        for name in self.direct_expert_names:
            key = name.lower()
            if key in ("mlp", "feature", "feature_mlp"):
                direct_modules[name] = FeatureExpert(hidden_dim, proj_dim, dropout)
            elif key in ("gcn", "raw_gcn", "graph"):
                direct_modules[name] = RawGCNExpert(
                    hidden_dim,
                    proj_dim,
                    num_layers=gcn_layers,
                    dropout=dropout,
                    residual=gcn_residual,
                    norm=gcn_norm,
                )
            else:
                raise ValueError(f"Unsupported direct expert: {name}")
        self.direct_experts = nn.ModuleDict(direct_modules)
        self.gate = MLP([proj_dim * self.num_views, hidden_dim, self.num_views], dropout)
        self.view_classifiers = nn.ModuleList([MLP([proj_dim, proj_dim, num_classes], dropout) for _ in range(self.num_views)])
        self.reliability = MLP([proj_dim + num_classes + 3, hidden_dim, 1], dropout)
        self.classifier = MLP([proj_dim, proj_dim, num_classes], dropout)
        self.num_regimes = 0
        self.regime_router = None
        if regime_groups:
            masks = []
            for group in regime_groups:
                mask = torch.zeros(self.num_views, dtype=torch.float32)
                for idx in group:
                    idx = int(idx)
                    if idx < 0 or idx >= self.num_views:
                        raise ValueError(f"Regime group index {idx} out of range for {self.num_views} experts")
                    mask[idx] = 1.0
                if mask.sum() <= 0:
                    raise ValueError("Regime groups must not be empty")
                masks.append(mask)
            regime_mask = torch.stack(masks, dim=0)
            self.num_regimes = int(regime_mask.shape[0])
            self.register_buffer("regime_mask", regime_mask, persistent=False)
            self.regime_router = MLP([proj_dim * self.num_views, hidden_dim, self.num_regimes], dropout)
            if regime_prior is None:
                self.register_buffer("regime_prior", torch.zeros(self.num_regimes), persistent=False)
            else:
                prior = torch.as_tensor(regime_prior, dtype=torch.float32)
                prior = prior / prior.sum().clamp_min(1e-12)
                self.register_buffer("regime_prior", torch.log(prior.clamp_min(1e-8)), persistent=False)
        else:
            self.register_buffer("regime_mask", torch.empty(0), persistent=False)
            self.register_buffer("regime_prior", torch.empty(0), persistent=False)
        if gate_prior is None:
            self.register_buffer("gate_prior", torch.empty(0), persistent=False)
        else:
            prior = torch.as_tensor(gate_prior, dtype=torch.float32)
            prior = prior / prior.sum().clamp_min(1e-12)
            self.register_buffer("gate_prior", torch.log(prior.clamp_min(1e-8)), persistent=False)

    def _compute_gate(self, z_list, view_logits_list=None):
        if self.gate_mode == "reliability":
            if view_logits_list is None:
                raise ValueError("reliability gate requires per-view logits")
            gate_z_list = [z.detach() for z in z_list] if self.gate_detach_inputs else z_list
            gate_logits_list = [logits.detach() for logits in view_logits_list] if self.gate_detach_inputs else view_logits_list
            prob_list = [F.softmax(logits, dim=-1) for logits in gate_logits_list]
            scores = []
            for idx, (z, probs) in enumerate(zip(gate_z_list, prob_list)):
                entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
                denom = torch.log(torch.tensor(float(probs.size(-1)), device=probs.device)).clamp_min(1e-8)
                norm_entropy = entropy / denom
                confidence = probs.max(dim=-1, keepdim=True).values
                if len(prob_list) > 1:
                    agree_terms = []
                    for j, other in enumerate(prob_list):
                        if j != idx:
                            agree_terms.append((probs * other).sum(dim=-1, keepdim=True))
                    agreement = torch.stack(agree_terms, dim=0).mean(dim=0)
                else:
                    agreement = torch.ones_like(confidence)
                feat = torch.cat([F.normalize(z, dim=-1), probs, norm_entropy, confidence, agreement], dim=-1)
                score = self.reliability(feat)
                if self.reliability_entropy_bias:
                    score = score - float(self.reliability_entropy_bias) * norm_entropy
                scores.append(score)
            logits = torch.cat(scores, dim=-1)
            if self.gate_prior.numel():
                logits = logits + self.gate_prior.view(1, -1)
            if self.regime_router is not None:
                gate, regime_probs, regime_logits = self._compute_regime_gate(logits, z_list)
                gate = self._postprocess_gate(gate)
                return gate, logits, regime_probs, regime_logits
            gate = F.softmax(logits / self.gate_temperature, dim=-1)
            if self.gate_residual > 0:
                uni = torch.full_like(gate, 1.0 / self.num_views)
                gate = (1 - self.gate_residual) * gate + self.gate_residual * uni
            gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            gate = self._postprocess_gate(gate)
            return gate, logits, None, None

        concat = torch.cat([F.normalize(z, dim=-1) for z in z_list], dim=-1)
        logits = self.gate(concat)
        if self.gate_prior.numel():
            logits = logits + self.gate_prior.view(1, -1)
        gate = F.softmax(logits / self.gate_temperature, dim=-1)
        if self.gate_residual > 0:
            uni = torch.full_like(gate, 1.0 / self.num_views)
            gate = (1 - self.gate_residual) * gate + self.gate_residual * uni
        gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        gate = self._postprocess_gate(gate)
        if self.gate_mode == "top2":
            val, idx = torch.topk(gate, k=min(2, self.num_views), dim=-1)
            masked = torch.zeros_like(gate)
            masked.scatter_(1, idx, val)
            gate = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return gate, logits, None, None

    def _compute_regime_gate(self, expert_logits, z_list):
        router_inputs = [z.detach() for z in z_list] if self.gate_detach_inputs else z_list
        concat = torch.cat([F.normalize(z, dim=-1) for z in router_inputs], dim=-1)
        regime_logits = self.regime_router(concat)
        if self.regime_prior.numel():
            regime_logits = regime_logits + self.regime_prior.view(1, -1)
        regime_probs = F.softmax(regime_logits / max(self.regime_temperature, 1e-6), dim=-1)
        if self.regime_sharpen_power != 1.0:
            regime_probs = regime_probs.clamp_min(1e-8).pow(self.regime_sharpen_power)
            regime_probs = regime_probs / regime_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if self.regime_topk and 0 < self.regime_topk < self.num_regimes:
            val, idx = torch.topk(regime_probs, k=self.regime_topk, dim=-1)
            masked = torch.zeros_like(regime_probs)
            masked.scatter_(1, idx, val)
            regime_probs = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        gates = []
        neg_inf = torch.finfo(expert_logits.dtype).min
        for group_idx in range(self.num_regimes):
            mask = self.regime_mask[group_idx].view(1, -1)
            masked_logits = expert_logits.masked_fill(mask <= 0, neg_inf)
            group_gate = F.softmax(masked_logits / max(self.gate_temperature, 1e-6), dim=-1)
            gates.append(group_gate)
        stacked = torch.stack(gates, dim=1)
        gate = (regime_probs.unsqueeze(-1) * stacked).sum(dim=1)
        if self.gate_residual > 0:
            uni = torch.full_like(gate, 1.0 / self.num_views)
            gate = (1 - self.gate_residual) * gate + self.gate_residual * uni
        gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return gate, regime_probs, regime_logits

    def _postprocess_gate(self, gate):
        if self.gate_sharpen_power != 1.0:
            gate = gate.clamp_min(1e-8).pow(self.gate_sharpen_power)
            gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if self.gate_adaptive_sparse and self.num_views > 1:
            val2, _ = torch.topk(gate, k=2, dim=-1)
            confident = (val2[:, 0:1] - val2[:, 1:2]) >= self.gate_adaptive_margin
            k = min(self.gate_adaptive_topk, self.num_views)
            val, idx = torch.topk(gate, k=k, dim=-1)
            masked = torch.zeros_like(gate)
            masked.scatter_(1, idx, val)
            sparse_gate = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            gate = torch.where(confident, sparse_gate, gate)
        elif self.gate_topk and 0 < self.gate_topk < self.num_views:
            val, idx = torch.topk(gate, k=self.gate_topk, dim=-1)
            masked = torch.zeros_like(gate)
            masked.scatter_(1, idx, val)
            gate = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return gate

    def forward(self, x, view_ops, adj=None):
        h = self.input_encoder(x)
        z_list, p_list, theta_list = [], [], []
        view_logits_list = []
        for idx, (enc, op) in enumerate(zip(self.views, view_ops)):
            z, p, theta = enc(op, h)
            z_list.append(z)
            p_list.append(p)
            theta_list.append(theta)
            view_logits_list.append(self.view_classifiers[idx](z))
        offset = len(z_list)
        for local_idx, name in enumerate(self.direct_expert_names):
            module = self.direct_experts[name]
            if name.lower() in ("gcn", "raw_gcn", "graph"):
                z, p = module(h, adj)
            else:
                z, p = module(h)
            z_list.append(z)
            p_list.append(p)
            theta_list.append(None)
            view_logits_list.append(self.view_classifiers[offset + local_idx](z))
        gate, gate_logits, regime_probs, regime_logits = self._compute_gate(z_list, view_logits_list)
        fused = torch.zeros_like(z_list[0])
        for i, z in enumerate(z_list):
            fused = fused + gate[:, i : i + 1] * z
        embed_logits = self.classifier(fused)
        if self.fusion_mode == "logits":
            final_logits = torch.zeros_like(view_logits_list[0])
            for i, logits in enumerate(view_logits_list):
                final_logits = final_logits + gate[:, i : i + 1] * logits
        elif self.fusion_mode == "hybrid":
            final_logits = embed_logits
            for i, logits in enumerate(view_logits_list):
                final_logits = final_logits + gate[:, i : i + 1] * logits
            final_logits = final_logits / 2.0
        else:
            final_logits = embed_logits
        return {
            "z_list": z_list,
            "p_list": p_list,
            "theta_list": theta_list,
            "view_logits_list": view_logits_list,
            "gate": gate,
            "gate_logits": gate_logits,
            "regime_probs": regime_probs,
            "regime_logits": regime_logits,
            "fused": fused,
            "embed_logits": embed_logits,
            "logits": final_logits,
        }


class RiskRouterHGNN(nn.Module):
    """Isolated path router for structural risk regimes.

    The model keeps feature, raw-structure, and view-risk paths separate. The
    final node prediction is routed across path logits, while the router can be
    trained with detached path logits to avoid flat-pool expert interference.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        proj_dim,
        num_classes,
        num_views=3,
        K=2,
        dropout=0.3,
        refine=False,
        feature_layers=4,
        structure_layers=2,
        structure_residual=True,
        structure_norm="layer",
        view_gate_temperature=1.0,
        view_gate_residual=0.0,
        view_gate_mode="reliability",
        view_fusion_mode="logits",
        view_gate_detach_inputs=True,
        view_gate_topk=0,
        view_gate_sharpen_power=1.0,
        view_gate_prior=None,
        view_reliability_entropy_bias=0.0,
        route_temperature=1.0,
        route_topk=0,
        route_sharpen_power=1.0,
        route_prior=None,
        route_detach_inputs=True,
        final_detach_paths=True,
        final_detach_mode=None,
        path_fusion="mixture",
        base_path="structure",
        residual_scale=0.5,
        utility_shared_stem=False,
        risk_feature_dim=0,
        path_logit_calibration=False,
        adaptive_calibration=False,
        calibration_temperature=1.0,
        adaptive_fusion=False,
        fusion_temperature=1.0,
        fusion_mode_prior=None,
    ):
        super().__init__()
        self.path_names = ["view_risk", "feature", "structure"]
        self.num_paths = len(self.path_names)
        self.num_views = int(num_views)
        self.num_classes = int(num_classes)
        self.view_gate_temperature = float(view_gate_temperature)
        self.view_gate_topk = int(view_gate_topk or 0)
        self.view_gate_sharpen_power = float(view_gate_sharpen_power)
        self.route_temperature = float(route_temperature)
        self.route_topk = int(route_topk or 0)
        self.route_sharpen_power = float(route_sharpen_power)
        self.route_detach_inputs = bool(route_detach_inputs)
        if final_detach_mode is None:
            self.final_detach_mode = "all" if bool(final_detach_paths) else "none"
        else:
            self.final_detach_mode = str(final_detach_mode).lower()
        self.path_fusion = str(path_fusion).lower()
        self.base_path = str(base_path)
        self.residual_scale = float(residual_scale)
        self.utility_shared_stem = bool(utility_shared_stem)
        self.risk_feature_dim = int(risk_feature_dim or 0)
        self.adaptive_calibration = bool(adaptive_calibration)
        self.path_logit_calibration = bool(path_logit_calibration or self.adaptive_calibration)
        self.calibration_temperature = float(calibration_temperature)
        self.adaptive_fusion = bool(adaptive_fusion)
        self.fusion_temperature = float(fusion_temperature)

        self.view_path = MVHGCL(
            in_dim,
            hidden_dim,
            proj_dim,
            num_classes,
            num_views=num_views,
            K=K,
            dropout=dropout,
            gate_temperature=view_gate_temperature,
            gate_residual=float(view_gate_residual),
            refine=refine,
            gate_mode=str(view_gate_mode),
            gate_prior=view_gate_prior,
            reliability_entropy_bias=view_reliability_entropy_bias,
            fusion_mode=str(view_fusion_mode),
            gate_detach_inputs=bool(view_gate_detach_inputs),
            gate_topk=view_gate_topk,
            gate_sharpen_power=view_gate_sharpen_power,
            gate_adaptive_sparse=False,
            direct_experts=None,
        )
        self.view_path_rng_state_after_init = torch.get_rng_state()

        if self.utility_shared_stem:
            self.utility_input = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout),
            )
            self.feature_path = FeatureExpert(hidden_dim, proj_dim, dropout)
            self.structure_path = RawGCNExpert(
                hidden_dim,
                proj_dim,
                num_layers=structure_layers,
                dropout=dropout,
                residual=structure_residual,
                norm=structure_norm,
            )
            self.utility_classifiers = nn.ModuleList(
                [MLP([proj_dim, proj_dim, num_classes], dropout) for _ in range(2)]
            )
        else:
            self.feature_path = FeatureMLP(
                in_dim,
                hidden_dim,
                num_classes,
                num_layers=feature_layers,
                dropout=dropout,
                layer_norm=True,
            )
            self.structure_path = DeepGCN(
                in_dim,
                hidden_dim,
                num_classes,
                num_layers=structure_layers,
                dropout=dropout,
                residual=structure_residual,
                norm=structure_norm,
            )
            self.utility_classifiers = None

        route_in_dim = self.num_paths * (num_classes + 3) + self.risk_feature_dim
        self.router = MLP([route_in_dim, hidden_dim, hidden_dim, self.num_paths], dropout)
        if self.path_logit_calibration:
            self.path_logit_log_scale = nn.Parameter(torch.zeros(self.num_paths))
            self.path_logit_bias = nn.Parameter(torch.zeros(self.num_paths, self.num_classes))
        else:
            self.register_parameter("path_logit_log_scale", None)
            self.register_parameter("path_logit_bias", None)
        if self.adaptive_calibration:
            self.calibration_controller = MLP([route_in_dim, hidden_dim, hidden_dim, self.num_paths], dropout)
        else:
            self.calibration_controller = None
        if self.adaptive_fusion:
            self.fusion_mode_names = ["top1", "top2", "soft"]
            self.fusion_controller = MLP([route_in_dim, hidden_dim, hidden_dim, len(self.fusion_mode_names)], dropout)
            if fusion_mode_prior is None:
                self.register_buffer("fusion_mode_prior", torch.empty(0), persistent=False)
            else:
                prior = torch.as_tensor(fusion_mode_prior, dtype=torch.float32)
                prior = prior / prior.sum().clamp_min(1e-12)
                self.register_buffer("fusion_mode_prior", torch.log(prior.clamp_min(1e-8)), persistent=False)
        else:
            self.fusion_mode_names = []
            self.fusion_controller = None
            self.register_buffer("fusion_mode_prior", torch.empty(0), persistent=False)
        if route_prior is None:
            self.register_buffer("route_prior", torch.empty(0), persistent=False)
        else:
            prior = torch.as_tensor(route_prior, dtype=torch.float32)
            prior = prior / prior.sum().clamp_min(1e-12)
            self.register_buffer("route_prior", torch.log(prior.clamp_min(1e-8)), persistent=False)

    def _topk_normalize(self, probs, topk):
        if topk and 0 < topk < probs.size(1):
            val, idx = torch.topk(probs, k=topk, dim=1)
            masked = torch.zeros_like(probs)
            masked.scatter_(1, idx, val)
            probs = masked / masked.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return probs

    def _route_features(self, path_logits_list, risk_features=None):
        logits_list = [logits.detach() for logits in path_logits_list] if self.route_detach_inputs else path_logits_list
        probs = [F.softmax(logits, dim=-1) for logits in logits_list]
        feats = []
        for idx, prob in enumerate(probs):
            entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
            denom = torch.log(torch.tensor(float(prob.size(-1)), device=prob.device)).clamp_min(1e-8)
            norm_entropy = entropy / denom
            confidence = prob.max(dim=-1, keepdim=True).values
            if len(probs) > 1:
                agreement = torch.stack(
                    [(prob * other).sum(dim=-1, keepdim=True) for j, other in enumerate(probs) if j != idx],
                    dim=0,
                ).mean(dim=0)
            else:
                agreement = torch.ones_like(confidence)
            feats.append(torch.cat([prob, norm_entropy, confidence, agreement], dim=-1))
        route_feat = torch.cat(feats, dim=-1)
        if self.risk_feature_dim:
            if risk_features is None:
                risk_features = route_feat.new_zeros(route_feat.size(0), self.risk_feature_dim)
            if risk_features.size(1) != self.risk_feature_dim:
                raise ValueError(f"Expected {self.risk_feature_dim} risk features, got {risk_features.size(1)}")
            route_feat = torch.cat([route_feat, risk_features.to(route_feat.device, route_feat.dtype)], dim=-1)
        return route_feat

    def _route_from_features(self, route_feat, apply_topk=True):
        route_logits = self.router(route_feat)
        if self.route_prior.numel():
            route_logits = route_logits + self.route_prior.view(1, -1)
        route_probs = F.softmax(route_logits / max(self.route_temperature, 1e-6), dim=-1)
        if self.route_sharpen_power != 1.0:
            route_probs = route_probs.clamp_min(1e-8).pow(self.route_sharpen_power)
            route_probs = route_probs / route_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if apply_topk:
            route_probs = self._topk_normalize(route_probs, self.route_topk)
        return route_probs, route_logits

    def _route(self, path_logits_list, risk_features=None, apply_topk=True):
        return self._route_from_features(
            self._route_features(path_logits_list, risk_features=risk_features),
            apply_topk=apply_topk,
        )

    def _calibrate_path_logits(self, path_logits_list):
        if not self.path_logit_calibration:
            return path_logits_list
        scales = self.path_logit_log_scale.exp()
        return [
            logits * scales[idx] + self.path_logit_bias[idx].view(1, -1)
            for idx, logits in enumerate(path_logits_list)
        ]

    def _adaptive_calibrate(self, raw_path_logits_list, risk_features=None):
        calibrated = self._calibrate_path_logits(raw_path_logits_list)
        if not self.adaptive_calibration:
            return calibrated, None
        raw_route_feat = self._route_features(raw_path_logits_list, risk_features=risk_features)
        alpha = torch.sigmoid(self.calibration_controller(raw_route_feat) / max(self.calibration_temperature, 1e-6))
        blended = [
            raw * (1.0 - alpha[:, idx : idx + 1]) + calibrated[idx] * alpha[:, idx : idx + 1]
            for idx, raw in enumerate(raw_path_logits_list)
        ]
        return blended, alpha

    def _adaptive_route_probs(self, route_feat):
        route_probs, route_logits = self._route_from_features(route_feat, apply_topk=False)
        # Risk pruning comes before fusion-mode selection. Otherwise the
        # adaptive controller can collapse to soft mixing and reintroduce paths
        # that should have been rejected, which is exactly the flat-pool failure
        # mode this architecture is meant to avoid.
        route_probs = self._topk_normalize(route_probs, self.route_topk)
        if not self.adaptive_fusion:
            return route_probs, route_logits, None
        mode_logits = self.fusion_controller(route_feat)
        if self.fusion_mode_prior.numel():
            mode_logits = mode_logits + self.fusion_mode_prior.view(1, -1)
        mode_probs = F.softmax(mode_logits / max(self.fusion_temperature, 1e-6), dim=-1)
        variants = [
            self._topk_normalize(route_probs, 1),
            self._topk_normalize(route_probs, min(2, route_probs.size(1))),
            route_probs,
        ]
        final_probs = torch.zeros_like(route_probs)
        for idx, probs in enumerate(variants):
            final_probs = final_probs + mode_probs[:, idx : idx + 1] * probs
        return final_probs, route_logits, mode_probs

    def forward(self, x, view_ops, adj=None, risk_features=None):
        view_out = self.view_path(x, view_ops, None)
        view_risk_logits = view_out["logits"]
        if self.utility_shared_stem:
            utility_h = self.utility_input(x)
            feature_z, _ = self.feature_path(utility_h)
            structure_z, _ = self.structure_path(utility_h, adj)
            feature_logits = self.utility_classifiers[0](feature_z)
            structure_logits = self.utility_classifiers[1](structure_z)
        else:
            feature_logits = self.feature_path(x)
            structure_logits = self.structure_path(x, adj)
        raw_path_logits_list = [view_risk_logits, feature_logits, structure_logits]
        path_logits_list, calibration_alpha = self._adaptive_calibrate(raw_path_logits_list, risk_features=risk_features)
        route_feat = self._route_features(path_logits_list, risk_features=risk_features)
        route_probs, route_logits, fusion_mode_probs = self._adaptive_route_probs(route_feat)

        if self.final_detach_mode in ("all", "true", "yes"):
            mix_logits = [logits.detach() for logits in path_logits_list]
        elif self.final_detach_mode in ("view_risk", "risk", "risk_only", "view_risk_only"):
            mix_logits = [path_logits_list[0].detach(), path_logits_list[1], path_logits_list[2]]
        elif self.final_detach_mode in ("utility", "utility_only"):
            mix_logits = [path_logits_list[0], path_logits_list[1].detach(), path_logits_list[2].detach()]
        elif self.final_detach_mode in ("none", "false", "no"):
            mix_logits = path_logits_list
        else:
            raise ValueError(f"Unsupported final_detach_mode: {self.final_detach_mode}")
        if self.path_fusion == "residual":
            base_idx = self.path_names.index(self.base_path) if self.base_path in self.path_names else 0
            base_logits = mix_logits[base_idx]
            residual = torch.zeros_like(base_logits)
            for idx, logits in enumerate(mix_logits):
                residual = residual + route_probs[:, idx : idx + 1] * (logits - base_logits)
            final_logits = base_logits + self.residual_scale * residual
        else:
            final_logits = torch.zeros_like(path_logits_list[0])
            for idx, logits in enumerate(mix_logits):
                final_logits = final_logits + route_probs[:, idx : idx + 1] * logits

        return {
            "z_list": view_out["z_list"],
            "p_list": view_out["p_list"],
            "theta_list": view_out["theta_list"],
            "view_logits_list": view_out["view_logits_list"],
            "path_logits_list": path_logits_list,
            "raw_path_logits_list": raw_path_logits_list,
            "path_names": self.path_names,
            "calibration_alpha": calibration_alpha,
            "fusion_mode_probs": fusion_mode_probs,
            "fusion_mode_names": self.fusion_mode_names,
            "gate": view_out["gate"],
            "gate_logits": view_out["gate_logits"],
            "route_probs": route_probs,
            "route_logits": route_logits,
            "regime_probs": route_probs,
            "regime_logits": route_logits,
            "logits": final_logits,
        }
