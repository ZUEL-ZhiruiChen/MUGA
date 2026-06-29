from __future__ import annotations

from itertools import combinations

import torch
import torch.nn.functional as F


def bootstrap_alignment(z_list, p_list):
    if len(z_list) < 2:
        return z_list[0].new_tensor(0.0)
    losses = []
    for i, j in combinations(range(len(z_list)), 2):
        pi = F.normalize(p_list[i], dim=-1)
        pj = F.normalize(p_list[j], dim=-1)
        zi = F.normalize(z_list[i].detach(), dim=-1)
        zj = F.normalize(z_list[j].detach(), dim=-1)
        losses.append(2 - 2 * (pi * zj).sum(dim=-1).mean())
        losses.append(2 - 2 * (pj * zi).sum(dim=-1).mean())
    return sum(losses) / len(losses)


def dominant_view_distill(z_list, gate, temperature=1.0, hard=False):
    if len(z_list) < 2:
        return z_list[0].new_tensor(0.0)
    if hard:
        with torch.no_grad():
            dom = gate.argmax(dim=-1)
        losses = []
        for v, z in enumerate(z_list):
            teacher = torch.stack([z_list[int(d)][i] for i, d in enumerate(dom)], dim=0).detach()
            mask = (dom != v).float()
            if mask.sum() > 0:
                loss = 2 - 2 * (F.normalize(z, dim=-1) * F.normalize(teacher, dim=-1)).sum(dim=-1)
                losses.append((loss * mask).sum() / mask.sum())
        return sum(losses) / len(losses) if losses else gate.new_tensor(0.0)
    with torch.no_grad():
        weights = F.softmax(torch.log(gate.clamp_min(1e-8)) / temperature, dim=-1)
        teacher_sum = sum(weights[:, i : i + 1] * z_list[i].detach() for i in range(len(z_list)))
    losses = []
    for v, z in enumerate(z_list):
        other_weight = (1.0 - weights[:, v]).clamp_min(1e-6)
        teacher = (teacher_sum - weights[:, v : v + 1] * z_list[v].detach()) / other_weight.unsqueeze(-1)
        loss = 2 - 2 * (F.normalize(z, dim=-1) * F.normalize(teacher, dim=-1)).sum(dim=-1)
        losses.append((loss * other_weight).sum() / other_weight.sum().clamp_min(1e-6))
    return sum(losses) / len(losses)


def variance_loss(z_list, gamma=1.0, eps=1e-4):
    if not z_list:
        return torch.tensor(0.0)
    vals = []
    for z in z_list:
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        vals.append(F.relu(gamma - std).mean())
    return sum(vals) / len(vals)


def elastic_interval_loss(embedding, pair_index, lower, upper):
    if pair_index.numel() == 0:
        return embedding.new_tensor(0.0)
    src, dst = pair_index[0], pair_index[1]
    e1 = F.normalize(embedding[src], dim=-1)
    e2 = F.normalize(embedding[dst], dim=-1)
    d = torch.clamp(1.0 - (e1 * e2).sum(dim=-1), 0.0, 2.0)
    return (F.relu(lower - d) + F.relu(d - upper)).mean()


def discriminative_elastic_loss(embedding, pair_index, lower, upper, y, train_mask, neg_margin=0.8):
    if pair_index.numel() == 0:
        return embedding.new_tensor(0.0)
    src, dst = pair_index[0], pair_index[1]
    e1 = F.normalize(embedding[src], dim=-1)
    e2 = F.normalize(embedding[dst], dim=-1)
    d = torch.clamp(1.0 - (e1 * e2).sum(dim=-1), 0.0, 2.0)
    pos = (F.relu(lower - d) + F.relu(d - upper)).mean()
    mask = train_mask[src] & train_mask[dst] & (y[src] != y[dst])
    neg = F.relu(neg_margin - d[mask]).mean() if mask.any() else embedding.new_tensor(0.0)
    return pos + neg


def load_balance_loss(gate):
    target = torch.full((gate.size(1),), 1.0 / gate.size(1), device=gate.device)
    avg = gate.mean(dim=0)
    return ((avg - target) ** 2).mean()
