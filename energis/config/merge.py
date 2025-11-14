from __future__ import annotations
import os, copy, hashlib, yaml
from typing import List, Dict, Any

def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_and_merge(paths: List[str]) -> Dict[str,Any]:
    cfg = {}
    norm = []
    for p in paths:
        if p is None: 
            continue
        p = os.path.abspath(p)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Config not found: {p}")
        norm.append(p)
        cfg = _deep_merge(cfg, load_yaml(p))
    # compute hash for provenance
    h = hashlib.sha256()
    for p in norm:
        with open(p, "rb") as f:
            h.update(f.read())
    cfg["meta"] = cfg.get("meta", {})
    cfg["meta"]["config_hash"] = h.hexdigest()[:16]
    cfg["meta"]["merged_from"] = norm
    return cfg
