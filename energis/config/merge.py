from __future__ import annotations

import copy
import hashlib
from pathlib import Path
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_config_path(path: str) -> Path:
    """Resolve ``path`` to an existing configuration file.

    Relative paths are first resolved against the current working directory
    (allowing callers to keep the old behaviour).  If that fails we fall back
    to the project root so that notebooks or scripts executed from nested
    directories can continue to refer to configs using repository-relative
    paths.
    """

    candidate = Path(path)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Config not found: {candidate}")

    search_roots = [Path.cwd()]
    if PROJECT_ROOT not in search_roots:
        search_roots.append(PROJECT_ROOT)

    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(f"Config not found: {(Path.cwd() / candidate).resolve()}")


def load_and_merge(paths: List[str]) -> Dict[str,Any]:
    cfg = {}
    norm = []
    for p in paths:
        if p is None:
            continue
        resolved = _resolve_config_path(p)
        norm.append(str(resolved))
        cfg = _deep_merge(cfg, load_yaml(str(resolved)))
    # compute hash for provenance
    h = hashlib.sha256()
    for p in norm:
        with open(p, "rb") as f:
            h.update(f.read())
    cfg["meta"] = cfg.get("meta", {})
    cfg["meta"]["config_hash"] = h.hexdigest()[:16]
    cfg["meta"]["merged_from"] = norm
    return cfg
