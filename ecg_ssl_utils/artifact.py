"""Write config + seed + git hash snapshots into pipeline output directories."""

import json
import os
import subprocess
import hashlib
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


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model_manifest(model_dirs: Dict[str, str], required_seeds) -> Dict[str, Any]:
    """Validate model identity and seed completeness before downstream use."""
    rows = []
    primary_prefixes = (
        "ssl-simclr-", "ssl-clocs-vit", "ssl-mae-", "ssl-jepa-",
        "ssl-supervised-",
    )
    for name, directory in sorted(model_dirs.items()):
        encoder_path = os.path.join(directory, "encoder.pt")
        config_path = os.path.join(directory, "config.json")
        if not os.path.exists(encoder_path) or not os.path.exists(config_path):
            raise FileNotFoundError(f"incomplete model artifact: {directory}")
        with open(config_path) as f:
            model_cfg = json.load(f)
        seed = model_cfg.get("seed", model_cfg.get("extra", {}).get("seed"))
        rows.append({
            "name": name,
            "seed": seed,
            "primary": any(name.startswith(p) for p in primary_prefixes),
            "encoder_sha256": file_sha256(encoder_path),
            "config_sha256": file_sha256(config_path),
            "git_commit": model_cfg.get("git_commit"),
            "package_version": model_cfg.get("package_version"),
        })

    required = set(int(s) for s in required_seeds)
    expected_stems = {
        "ssl-simclr-vit-small", "ssl-clocs-vit-small", "ssl-mae-vit-small",
        "ssl-jepa-vit-small", "ssl-supervised-vit-small",
    }
    by_stem: Dict[str, set] = {}
    for row in rows:
        if not row["primary"]:
            continue
        stem = row["name"].split("-seed")[0]
        by_stem.setdefault(stem, set()).add(int(row["seed"]))
    incomplete = {
        stem: sorted(required - by_stem.get(stem, set()))
        for stem in expected_stems if required - by_stem.get(stem, set())
    }
    if incomplete:
        raise RuntimeError(f"incomplete primary pretraining seed grid: {incomplete}")
    return {
        "models": rows,
        "required_primary_seeds": sorted(required),
        "cohort_definition": "official_all_diagnostic_statements",
    }
