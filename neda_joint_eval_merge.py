#!/usr/bin/env python3
"""Fail-closed merge for sharded joint-policy ALFWorld evaluation."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from neda_freeze_splits import scan_games
from neda_repro import load_split_manifest, sha256_file, sha256_json


CONTRACT_VERSION = "neda-joint-alfworld-evaluation-v1"


def _atomic_json(payload: Dict[str, Any], path: str) -> None:
    path = os.path.realpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument(
        "--split-name", choices=("dev_seen", "final_unseen"), required=True
    )
    parser.add_argument("--method", choices=("neda", "mapg", "dcolt"), required=True)
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--expected-games", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = load_split_manifest(args.split_manifest)
    split = manifest["splits"][args.split_name]
    expected_ids = {
        str(row["game_id"]) for row in scan_games(split["source_root_hint"])
    }
    if len(expected_ids) != int(split["n_games"]):
        raise ValueError("frozen split inventory count drift")
    if int(args.expected_games) != len(expected_ids):
        raise ValueError("requested evaluation count is not the complete frozen split")

    rows: List[Dict[str, Any]] = []
    receipts = []
    identities = set()
    head_hashes = set()
    for raw_path in args.inputs:
        path = os.path.realpath(raw_path)
        with open(path, "r", encoding="utf-8") as handle:
            shard = json.load(handle)
        if shard.get("partial"):
            raise ValueError("evaluation shard is incomplete")
        if shard.get("split_name") != args.split_name:
            raise ValueError("evaluation split drift")
        if shard.get("rl_method") != args.method:
            raise ValueError("evaluation method drift")
        if int(shard.get("eval_seed", -1)) != int(args.eval_seed):
            raise ValueError("evaluation seed drift")
        if abs(float(shard.get("position_temperature")) - 0.5) > 1e-12:
            raise ValueError("evaluation position temperature drift")
        identity = shard.get("model_identity", {}).get("identity_sha256")
        if not identity:
            raise ValueError("evaluation shard lacks model identity")
        identities.add(str(identity))
        head_hash = shard.get("dcolt_head_sha256")
        if args.method == "dcolt" and not head_hash:
            raise ValueError("DCoLT evaluation omitted its trained UPM identity")
        if args.method != "dcolt" and head_hash is not None:
            raise ValueError("non-DCoLT evaluation unexpectedly has a UPM")
        head_hashes.add(head_hash)
        shard_rows = shard.get("results")
        if (
            not isinstance(shard_rows, list)
            or len(shard_rows) != int(shard.get("num_games", -1))
        ):
            raise ValueError("evaluation shard result count drift")
        rows.extend(dict(row) for row in shard_rows)
        receipts.append(
            {
                "path": path,
                "sha256": sha256_file(path),
                "offset": int(shard.get("offset", -1)),
                "n_games": len(shard_rows),
            }
        )

    if len(identities) != 1 or len(head_hashes) != 1:
        raise ValueError("evaluation shards used different policy identities")
    observed_ids = [str(row.get("game_id", "")) for row in rows]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("evaluation shards overlap in frozen tasks")
    if set(observed_ids) != expected_ids:
        missing = sorted(expected_ids - set(observed_ids))
        extra = sorted(set(observed_ids) - expected_ids)
        raise ValueError(
            "frozen evaluation coverage drift missing={} extra={}".format(
                missing[:5], extra[:5]
            )
        )
    rows.sort(key=lambda row: str(row["game_id"]))

    turns = [
        turn
        for row in rows
        for turn in row.get("trajectory", [])
    ]
    task_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_groups[str(row["game_id"]).split("-", 1)[0]].append(row)
    by_task_type = {
        name: {
            "n": len(values),
            "successes": sum(bool(row["success"]) for row in values),
            "success_rate": (
                sum(bool(row["success"]) for row in values) / len(values)
            ),
            "mean_progress": (
                sum(float(row["progress"]) for row in values) / len(values)
            ),
        }
        for name, values in sorted(task_groups.items())
    }
    successes = sum(bool(row["success"]) for row in rows)
    payload = {
        "contract_version": CONTRACT_VERSION,
        "status": "PASS",
        "scientific_result": True,
        "environment": "alfworld",
        "method": args.method,
        "eval_seed": int(args.eval_seed),
        "split_name": args.split_name,
        "split_manifest": os.path.realpath(args.split_manifest),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "model_identity_sha256": next(iter(identities)),
        "dcolt_head_sha256": next(iter(head_hashes)),
        "n_games": len(rows),
        "successes": successes,
        "success_at_1": successes / len(rows),
        "mean_progress": sum(float(row["progress"]) for row in rows) / len(rows),
        "raw_legality_rate": (
            sum(bool(turn["raw_is_legal"]) for turn in turns) / len(turns)
            if turns else 0.0
        ),
        "sent_legality_rate": (
            sum(bool(turn["sent_is_legal"]) for turn in turns) / len(turns)
            if turns else 0.0
        ),
        "environment_interactions": len(turns),
        "mean_horizon": len(turns) / len(rows),
        "by_task_type": by_task_type,
        "input_receipts": sorted(receipts, key=lambda value: value["offset"]),
        "results": rows,
    }
    body = dict(payload)
    payload["artifact_sha256"] = sha256_json(body)
    _atomic_json(payload, args.out)
    print(
        json.dumps(
            {
                "status": "PASS",
                "method": args.method,
                "n_games": len(rows),
                "successes": successes,
                "success_at_1": payload["success_at_1"],
                "out": os.path.realpath(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
