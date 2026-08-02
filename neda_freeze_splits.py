#!/usr/bin/env python3
"""Create or verify the frozen ALFWorld dev/final split manifest.

The manifest stores a hash over sorted machine-independent game IDs instead of
absolute paths. Runtime code must rescan the configured root and pass the hash
check before evaluation, so an accidental dataset change fails closed.
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

from neda_repro import (
    SPLIT_CONTRACT_VERSION,
    canonical_game_id,
    check_game_ids,
    load_split_manifest,
    sha256_json,
)


ALLOWED_TASK_TYPES = {
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
}


def scan_games(root) -> List[Dict[str, Any]]:
    root = Path(os.path.expandvars(os.path.expanduser(str(root)))).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ALFWorld split root does not exist: {root}")
    games: List[Dict[str, Any]] = []
    for game_file in sorted(root.rglob("game.tw-pddl")):
        directory_text = str(game_file.parent)
        if "movable" in directory_text or "Sliced" in directory_text:
            continue
        traj_path = game_file.with_name("traj_data.json")
        if not traj_path.is_file():
            continue
        with open(traj_path, "r", encoding="utf-8") as handle:
            trajectory = json.load(handle)
        task_type = trajectory.get("task_type")
        if task_type not in ALLOWED_TASK_TYPES:
            continue
        with open(game_file, "r", encoding="utf-8") as handle:
            game_data = json.load(handle)
        if not game_data.get("solvable", False):
            continue
        games.append(
            {
                "game_id": canonical_game_id(game_file),
                "task_type": task_type,
                "path": str(game_file),
            }
        )
    ids = [row["game_id"] for row in games]
    if ids != sorted(ids):
        raise AssertionError("scanner must return sorted canonical IDs")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate canonical IDs below {root}")
    return games


def split_summary(root: str, role: str, exposed: bool) -> Dict[str, Any]:
    games = scan_games(root)
    game_ids = [row["game_id"] for row in games]
    return {
        "role": role,
        "historically_exposed": bool(exposed),
        "source_root_hint": root,
        "selection_rule": "ALFWorld task_types 1-6; solvable; exclude movable/Sliced; canonical sort",
        "n_games": len(game_ids),
        "game_ids_sha256": sha256_json(game_ids),
        "task_type_counts": dict(sorted(Counter(row["task_type"] for row in games).items())),
    }


def verify_manifest(manifest_path: str, roots: Mapping[str, str]) -> None:
    manifest = load_split_manifest(manifest_path)
    for name, root in roots.items():
        spec = manifest["splits"][name]
        game_ids = [row["game_id"] for row in scan_games(root)]
        check_game_ids(game_ids, spec)
        print(f"[split] {name}: OK n={len(game_ids)} hash={spec['game_ids_sha256'][:12]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev-root",
        default="${ALFWORLD_DATA:-$HOME/.cache/alfworld}/json_2.1.1/valid_seen",
    )
    parser.add_argument(
        "--final-root",
        default="${ALFWORLD_DATA:-$HOME/.cache/alfworld}/json_2.1.1/valid_unseen",
    )
    parser.add_argument("--out", help="write a new manifest; omit with --verify")
    parser.add_argument("--verify", help="verify an existing manifest")
    args = parser.parse_args()

    # ``expandvars`` cannot evaluate shell ``:-`` syntax; defaults use the
    # explicit cache path unless the caller supplies roots.
    cache_root = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
    dev_root = args.dev_root.replace(
        "${ALFWORLD_DATA:-$HOME/.cache/alfworld}", cache_root
    )
    final_root = args.final_root.replace(
        "${ALFWORLD_DATA:-$HOME/.cache/alfworld}", cache_root
    )
    roots = {"dev_seen": dev_root, "final_unseen": final_root}

    if args.verify:
        verify_manifest(args.verify, roots)
        return
    if not args.out:
        parser.error("provide --out to create a manifest, or --verify to verify one")
    manifest = {
        "contract_version": SPLIT_CONTRACT_VERSION,
        "frozen_at": "2026-07-15",
        "policy": {
            "model_selection_split": "dev_seen",
            "locked_confirmation_split": "final_unseen",
            "final_access": "only after method/config freeze",
            "caveat": "final_unseen was exposed to legacy runs; require a second environment or untouched task seeds for pristine confirmation",
        },
        "splits": {
            "dev_seen": split_summary(dev_root, "development/model selection", False),
            "final_unseen": split_summary(final_root, "locked legacy confirmation", True),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[split] wrote {out} hash={sha256_json(manifest)[:12]}")
    verify_manifest(str(out), roots)


if __name__ == "__main__":
    main()
