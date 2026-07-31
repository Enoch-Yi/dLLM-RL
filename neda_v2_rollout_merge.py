#!/usr/bin/env python3
"""Fail-closed NeDA-v2 wrapper around the frozen ALFWorld shard merger."""

from __future__ import annotations

import argparse
import json
import os

from neda_repro import sha256_file
from neda_v2_decision import NEDA_V2_EVIDENCE_CONTRACT_VERSION
from neda_v4_rollout_merge_sft import _atomic_json, merge_rollouts


NEDA_V2_MERGE_CONTRACT_VERSION = "neda-v2-alfworld-rollout-merge-v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--expected-games", type=int, required=True)
    parser.add_argument("--expected-group-size", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    for path in args.inputs:
        manifest_path = os.path.realpath(path) + ".manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            manifest.get("rl_variant") != "neda_v2"
            or manifest.get("credit_evidence_contract")
            != NEDA_V2_EVIDENCE_CONTRACT_VERSION
        ):
            raise ValueError("rollout shard lacks NeDA-v2 provenance")

    rows, manifest = merge_rollouts(
        args.inputs, args.expected_games, args.expected_group_size
    )
    for row in rows:
        thought = row.get("decision_traces", {}).get("thought", {})
        evidence = thought.get("action_evidence_v2", {})
        if (
            evidence.get("contract_version")
            != NEDA_V2_EVIDENCE_CONTRACT_VERSION
        ):
            raise ValueError("rollout row lacks NeDA-v2 legal-Action evidence")
    _atomic_json(rows, args.out)
    manifest.update(
        {
            "merge_contract_version": NEDA_V2_MERGE_CONTRACT_VERSION,
            "rl_variant": "neda_v2",
            "credit_evidence_contract": NEDA_V2_EVIDENCE_CONTRACT_VERSION,
            "data_path": os.path.realpath(args.out),
            "data_sha256": sha256_file(args.out),
        }
    )
    _atomic_json(manifest, args.out + ".manifest.json")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
