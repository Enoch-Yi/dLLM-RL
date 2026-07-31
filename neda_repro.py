"""Reproducibility primitives shared by NeDA rollout, training, and eval.

This module intentionally has no mandatory third-party dependencies so split and
seed contracts can be checked on login nodes without importing PyTorch/ALFWorld.
"""

import hashlib
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Union


SEED_CONTRACT_VERSION = "neda-seeds-v1"
SPLIT_CONTRACT_VERSION = "neda-alfworld-splits-v1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Union[os.PathLike, str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_model_identity_payload(identity: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip node-local mount aliases from a v1/v2 model identity.

    RAIDEN exposes the same home filesystem as ``/home`` in a dev QRSH and as
    ``/uge_mnt/home`` in batch containers.  Runtime paths remain useful
    provenance, but cannot define checkpoint equality across those nodes.
    """

    keys = (
        "config_json_sha256",
        "generation_config_json_sha256",
        "model_safetensors_index_json_sha256",
        "model_class",
        "forward_source_sha256",
    )
    payload: Dict[str, Any] = {
        "contract_version": "neda-model-portable-identity-v1",
    }
    for key in keys:
        if key in identity:
            payload[key] = identity[key]
    payload["weight_shards"] = sorted(
        [
            {
                "name": str(shard["name"]),
                "size_bytes": int(shard["size_bytes"]),
            }
            for shard in identity.get("weight_shards", [])
        ],
        key=lambda shard: shard["name"],
    )
    return payload


def portable_model_identity_sha256(identity: Mapping[str, Any]) -> str:
    value = sha256_json(portable_model_identity_payload(identity))
    declared = identity.get("portable_identity_sha256")
    if declared is not None and str(declared) != value:
        raise ValueError("declared portable model identity SHA is inconsistent")
    return value


def model_identities_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return portable_model_identity_sha256(left) == portable_model_identity_sha256(right)


def build_model_identity(model_dir: Union[os.PathLike, str], model_class: Any = None) -> Dict[str, Any]:
    """Return a cheap, reproducible identity for the checkpoint and forward code.

    Weight shards are represented by their resolved path and byte size; hashing the
    multi-GB tensors for every rollout would dominate startup.  The weight index,
    config, and actual Python source used for the forward pass are content-hashed.
    """

    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {root}")
    payload: Dict[str, Any] = {
        "contract_version": "neda-model-identity-v2",
        "model_dir": str(root),
    }
    for name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
        path = root / name
        if path.is_file():
            payload[name.replace(".", "_") + "_sha256"] = sha256_file(path)

    source_path = None
    if model_class is not None:
        try:
            source_path = inspect.getsourcefile(model_class)
        except (TypeError, OSError):
            source_path = None
        payload["model_class"] = f"{model_class.__module__}.{model_class.__name__}"
    if source_path is None:
        candidate = root / "modeling_sdar.py"
        source_path = str(candidate) if candidate.is_file() else None
    if source_path and os.path.isfile(source_path):
        source = Path(source_path).resolve()
        payload["forward_source"] = str(source)
        payload["forward_source_sha256"] = sha256_file(source)

    index_path = root / "model.safetensors.index.json"
    shards = []
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        for name in sorted(set(index.get("weight_map", {}).values())):
            shard = root / name
            if not shard.exists():
                raise FileNotFoundError(f"model shard listed by index is missing: {shard}")
            resolved = shard.resolve()
            shards.append(
                {"name": name, "resolved_path": str(resolved), "size_bytes": resolved.stat().st_size}
            )
    payload["weight_shards"] = shards
    portable_sha = portable_model_identity_sha256(payload)
    payload["portable_identity_sha256"] = portable_sha
    # New artifacts use the portable value as their record-level policy ID.
    # The runtime paths above remain visible provenance but are not hashed.
    payload["identity_sha256"] = portable_sha
    return payload


def stable_seed(base_seed: int, *parts: object) -> int:
    """Derive an order-independent 31-bit seed from a named decision unit."""

    payload = [int(base_seed), *[str(part) for part in parts]]
    value = int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:8], "big")
    return value % (2**31 - 1)


def seed_everything(seed: int) -> None:
    """Seed Python and any already-installed NumPy/PyTorch RNGs.

    Optional imports keep the manifest/audit tools usable in a minimal Python
    environment. CUDA is only touched when it is available.
    """

    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed % (2**32 - 1))
    except ImportError:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def canonical_game_id(path: Union[os.PathLike, str]) -> str:
    """Return the machine-independent ``task/trial/game.tw-pddl`` identifier."""

    p = Path(path)
    if p.name != "game.tw-pddl":
        raise ValueError(f"not an ALFWorld game file: {path}")
    if len(p.parts) < 3:
        raise ValueError(f"cannot form canonical game id from: {path}")
    return "/".join(p.parts[-3:])


def load_json(path: Union[os.PathLike, str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_seed_manifest(path: Union[os.PathLike, str]) -> Dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("contract_version") != SEED_CONTRACT_VERSION:
        raise ValueError(
            f"seed manifest contract mismatch: {manifest.get('contract_version')!r}"
        )
    return manifest


def resolve_seed_bundle(
    manifest: Mapping[str, Any], phase: str, replicate: int = 0
) -> Dict[str, Any]:
    rows = manifest.get("phases", {}).get(phase)
    if not isinstance(rows, list) or not rows:
        raise KeyError(f"unknown/empty seed phase: {phase}")
    if replicate < 0 or replicate >= len(rows):
        raise IndexError(f"replicate {replicate} outside phase {phase} (n={len(rows)})")
    row = dict(rows[replicate])
    required = ("train_seed", "rollout_seed", "mask_seed", "sample_order_seed")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"seed row missing fields: {missing}")
    eval_seeds = row.get("eval_seeds")
    if not isinstance(eval_seeds, list) or not eval_seeds:
        raise ValueError("seed row requires a non-empty eval_seeds list")
    return row


def load_split_manifest(path: Union[os.PathLike, str]) -> Dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("contract_version") != SPLIT_CONTRACT_VERSION:
        raise ValueError(
            f"split manifest contract mismatch: {manifest.get('contract_version')!r}"
        )
    return manifest


def check_game_ids(game_ids: Sequence[str], split_spec: Mapping[str, Any]) -> None:
    expected_n = int(split_spec["n_games"])
    expected_hash = str(split_spec["game_ids_sha256"])
    actual = list(game_ids)
    if actual != sorted(actual):
        raise ValueError("game ids must be in canonical sorted order")
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate game ids in split")
    if len(actual) != expected_n:
        raise ValueError(f"split size mismatch: expected {expected_n}, got {len(actual)}")
    actual_hash = sha256_json(actual)
    if actual_hash != expected_hash:
        raise ValueError(
            f"split hash mismatch: expected {expected_hash}, got {actual_hash}"
        )


def order_game_files_by_manifest(
    game_files: Iterable[Union[os.PathLike, str]], game_ids: Sequence[str]
) -> List[str]:
    """Filter and reorder environment paths to the frozen canonical ID order."""

    by_id: Dict[str, str] = {}
    for path in game_files:
        game_id = canonical_game_id(path)
        if game_id in by_id:
            raise ValueError(f"duplicate environment game id: {game_id}")
        by_id[game_id] = str(path)
    missing = [game_id for game_id in game_ids if game_id not in by_id]
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(f"environment is missing {len(missing)} frozen games: {preview}")
    return [by_id[game_id] for game_id in game_ids]
