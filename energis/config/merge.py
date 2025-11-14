from __future__ import annotations

import os
import copy
import hashlib
from typing import List, Dict, Any

from energis.utils import simple_yaml

def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def load_yaml(path: str) -> dict:
    """Load YAML using the light-weight parser.

    PyYAML is not available in the execution environment.  The configuration
    files only rely on a tiny subset of the YAML specification, therefore we
    can use :mod:`energis.utils.simple_yaml` which is purposely written for this
    repository.
    """

    data = simple_yaml.load(path)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at document root in {path!r}.")
    return data

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
