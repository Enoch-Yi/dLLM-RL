#!/usr/bin/env python3
"""Fail-closed merge of independently generated ALFWorld online shards."""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from neda_data_contract import TRACE_CONTRACT_VERSION
from neda_repro import model_identities_match, sha256_file, sha256_json


MERGE_CONTRACT_VERSION = "neda-v4-online-rollout-merge-v1"
PROTOCOL_FIELDS = (
    "game_seed",
    "rollout_seed",
    "group_size",
    "max_steps",
    "gen_length",
    "block_length",
    "denoising_steps",
    "decision_decode",
    "thought_order",
    "action_order",
    "action_grammar",
    "execution_policy",
    "temperature",
    "online_credit_context_contract",
    "rl_method",
    "position_temperature",
    "dcolt_head_path",
    "dcolt_head_sha256",
    "dcolt_head_contract",
    "neda_credit_boundaries",
)


def _load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(value: Any, path: str) -> None:
    path = os.path.realpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_shard(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = os.path.realpath(path)
    rows = _load(path)
    manifest_path = path + ".manifest.json"
    manifest = _load(manifest_path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("rollout shard is empty: {}".format(path))
    if manifest.get("contract_version") != TRACE_CONTRACT_VERSION:
        raise ValueError("rollout shard trace contract drift")
    if manifest.get("data_sha256") != sha256_file(path):
        raise ValueError("rollout shard data SHA drift: {}".format(path))
    if int(manifest.get("n_records", -1)) != len(rows):
        raise ValueError("rollout shard record count drift")
    if manifest.get("online_credit_context_contract") != "neda-online-credit-state-v1":
        raise ValueError("rollout shard lacks online credit state")
    return [dict(row) for row in rows], manifest


def merge_rollouts(
    paths: Sequence[str], expected_games: int, expected_group_size: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if len(paths) != int(expected_games) or expected_games <= 0:
        raise ValueError("one complete shard is required per expected game")
    shards = [_read_shard(path) for path in paths]
    reference = shards[0][1]
    for _, manifest in shards[1:]:
        for field in PROTOCOL_FIELDS:
            if manifest.get(field) != reference.get(field):
                raise ValueError("rollout shard protocol drift: {}".format(field))
        if not model_identities_match(manifest["model_identity"], reference["model_identity"]):
            raise ValueError("rollout shard model identity drift")
    if int(reference.get("group_size", -1)) != int(expected_group_size):
        raise ValueError("rollout shard group-size drift")

    offsets = [int(manifest.get("offset", -1)) for _, manifest in shards]
    if any(offset < 0 for offset in offsets) or len(set(offsets)) != len(offsets):
        raise ValueError("rollout shard offsets must be nonnegative and unique")
    if any(int(manifest.get("num_games", -1)) != 1 for _, manifest in shards):
        raise ValueError("merged rollout requires exactly one game per shard")

    rows = [row for selected, _ in shards for row in selected]
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("merged rollout sample IDs are empty or duplicated")
    if any(row.get("raw_action") != row.get("executed_action") for row in rows):
        raise ValueError("merged rollout requires raw Action == executed Action")

    episodes: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    groups: Dict[str, set] = defaultdict(set)
    for row in rows:
        episodes[str(row["episode_id"])].append(row)
        groups[str(row["group_id"])].add(str(row["episode_id"]))
    if len(groups) != int(expected_games):
        raise ValueError("merged rollout game/group count drift")
    if any(len(values) != int(expected_group_size) for values in groups.values()):
        raise ValueError("merged rollout group is incomplete")
    for episode_id, selected in episodes.items():
        selected = sorted(selected, key=lambda row: int(row["turn_id"]))
        horizon = int(selected[0]["episode_horizon"])
        if len(selected) != horizon or [int(row["turn_id"]) for row in selected] != list(range(horizon)):
            raise ValueError("merged rollout episode is non-contiguous: {}".format(episode_id))
        if len({str(row["game_id"]) for row in selected}) != 1:
            raise ValueError("merged rollout episode crossed game IDs")

    rows.sort(
        key=lambda row: (
            str(row["game_id"]), int(row["rollout_id"]), int(row["turn_id"])
        )
    )
    returns_by_group = defaultdict(list)
    for selected in episodes.values():
        first = selected[0]
        returns_by_group[str(first["group_id"])].append(float(first["episode_return"]))
    signal_groups = sum(len(set(values)) > 1 for values in returns_by_group.values())
    source_receipts = [
        {
            "path": os.path.realpath(path),
            "sha256": sha256_file(path),
            "manifest_path": os.path.realpath(path) + ".manifest.json",
            "manifest_sha256": sha256_file(os.path.realpath(path) + ".manifest.json"),
            "offset": int(manifest["offset"]),
        }
        for path, (_, manifest) in zip(paths, shards)
    ]
    manifest = {
        "contract_version": TRACE_CONTRACT_VERSION,
        "merge_contract_version": MERGE_CONTRACT_VERSION,
        "status": "PASS",
        "n_records": len(rows),
        "n_games": len(groups),
        "n_episodes": len(episodes),
        "environment_interactions": len(rows),
        "n_signal_groups": signal_groups,
        "expected_group_size": int(expected_group_size),
        "source_receipts": source_receipts,
        "source_set_sha256": sha256_json(source_receipts),
        "sample_order_sha256": sha256_json([row["sample_id"] for row in rows]),
        "model_identity": reference["model_identity"],
    }
    for field in PROTOCOL_FIELDS:
        manifest[field] = reference.get(field)
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--expected-games", type=int, required=True)
    parser.add_argument("--expected-group-size", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows, manifest = merge_rollouts(
        args.inputs, args.expected_games, args.expected_group_size
    )
    _atomic_json(rows, args.out)
    manifest["data_path"] = os.path.realpath(args.out)
    manifest["data_sha256"] = sha256_file(args.out)
    _atomic_json(manifest, args.out + ".manifest.json")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
