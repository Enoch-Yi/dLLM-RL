#!/usr/bin/env python3
"""Fail-closed merge for NeDA-v2 WebShop rollout shards."""

from __future__ import annotations

import argparse
import json
import os

from neda_joint_webshop_rollout_merge import WEB_PROTOCOL_FIELDS
from neda_repro import sha256_file, sha256_json
from neda_v2_decision import NEDA_V2_EVIDENCE_CONTRACT_VERSION
from neda_v4_rollout_merge import _atomic_json, merge_rollouts


CONTRACT_VERSION = "neda-v2-webshop-rollout-merge-v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--expected-group-size", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifests = []
    for path in args.inputs:
        with open(path + ".manifest.json", encoding="utf-8") as handle:
            manifests.append(json.load(handle))
    reference = manifests[0]
    if reference.get("environment") != "webshop":
        raise ValueError("NeDA-v2 WebShop merger received another environment")
    for manifest in manifests:
        if (
            manifest.get("rl_method") != "neda"
            or manifest.get("rl_variant") != "neda_v2"
            or manifest.get("credit_evidence_contract")
            != NEDA_V2_EVIDENCE_CONTRACT_VERSION
        ):
            raise ValueError("WebShop shard lacks NeDA-v2 provenance")
        for field in WEB_PROTOCOL_FIELDS:
            if manifest.get(field) != reference.get(field):
                raise ValueError(
                    "NeDA-v2 WebShop shard protocol drift: {}".format(field)
                )

    task_ids = [
        int(manifest["task_ids"][0])
        for manifest in manifests
        if len(manifest.get("task_ids", [])) == 1
    ]
    if (
        len(task_ids) != int(args.expected_tasks)
        or len(task_ids) != len(set(task_ids))
    ):
        raise ValueError("NeDA-v2 WebShop task coverage/count drift")

    rows, manifest = merge_rollouts(
        args.inputs, args.expected_tasks, args.expected_group_size
    )
    for row in rows:
        evidence = (
            row.get("decision_traces", {})
            .get("thought", {})
            .get("action_evidence_v2", {})
        )
        if (
            evidence.get("contract_version")
            != NEDA_V2_EVIDENCE_CONTRACT_VERSION
        ):
            raise ValueError("WebShop row lacks legal-Action margin evidence")

    _atomic_json(rows, args.out)
    manifest.update(
        {
            "environment": "webshop",
            "merge_contract_version": CONTRACT_VERSION,
            "rl_variant": "neda_v2",
            "credit_evidence_contract": (
                NEDA_V2_EVIDENCE_CONTRACT_VERSION
            ),
            "webshop_task_ids": sorted(task_ids),
            "webshop_task_ids_sha256": sha256_json(sorted(task_ids)),
            "data_path": os.path.realpath(args.out),
            "data_sha256": sha256_file(args.out),
        }
    )
    for field in WEB_PROTOCOL_FIELDS:
        manifest[field] = reference.get(field)
    _atomic_json(manifest, args.out + ".manifest.json")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
