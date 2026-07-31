#!/usr/bin/env python3
"""Fail-closed WebShop wrapper around the shared online rollout merger."""

from __future__ import annotations

import argparse
import json
import os

from neda_repro import sha256_file, sha256_json
from neda_v4_rollout_merge import _atomic_json, merge_rollouts


WEB_PROTOCOL_FIELDS = (
    "environment",
    "rollout_contract_version",
    "train_seed",
    "action_gen_length",
    "game_seed",
    "split_manifest_sha256",
    "seed_manifest_sha256",
    "position_policy",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--expected-group-size", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    shard_manifests = []
    for path in args.inputs:
        with open(path + ".manifest.json", encoding="utf-8") as handle:
            shard_manifests.append(json.load(handle))
    reference = shard_manifests[0]
    if reference.get("environment") != "webshop":
        raise ValueError("WebShop merger received a foreign environment")
    for manifest in shard_manifests[1:]:
        for field in WEB_PROTOCOL_FIELDS:
            if manifest.get(field) != reference.get(field):
                raise ValueError(
                    "WebShop rollout shard protocol drift: {}".format(field)
                )
    task_ids = [
        int(manifest["task_ids"][0])
        for manifest in shard_manifests
        if len(manifest.get("task_ids", [])) == 1
    ]
    if (
        len(task_ids) != int(args.expected_tasks)
        or len(task_ids) != len(set(task_ids))
    ):
        raise ValueError("WebShop rollout task coverage/count drift")

    rows, manifest = merge_rollouts(
        args.inputs, args.expected_tasks, args.expected_group_size
    )
    manifest["environment"] = "webshop"
    manifest["webshop_task_ids"] = sorted(task_ids)
    manifest["webshop_task_ids_sha256"] = sha256_json(sorted(task_ids))
    for field in WEB_PROTOCOL_FIELDS:
        manifest[field] = reference.get(field)
    _atomic_json(rows, args.out)
    manifest["data_path"] = os.path.realpath(args.out)
    manifest["data_sha256"] = sha256_file(args.out)
    _atomic_json(manifest, args.out + ".manifest.json")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
