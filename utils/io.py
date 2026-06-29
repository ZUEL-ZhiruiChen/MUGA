from __future__ import annotations
import json, os, random
import numpy as np, torch

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
