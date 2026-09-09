"""Write config + seed + git hash snapshots into pipeline output directories."""

import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from .version import __version__


def _jsonify(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return out.decode().strip()
    except Exception:
        return None


def write_artifact_snapshot(
    output_dir: str,
    cfg: Any,
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    filename: str = "config.json",
) -> str:
    """Dump pipeline config, seed, package version, and git hash to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    payload: Dict[str, Any] = {
        "package_version": __version__,
        "git_commit": git_commit_hash(),
        "seed": seed,
        "config": _jsonify(cfg),
    }
    if extra:
        payload.update(_jsonify(extra))
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
