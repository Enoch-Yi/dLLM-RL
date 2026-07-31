#!/usr/bin/env python3
"""WebShop adapter for the NeDA-v2 legal-Action credit materializer."""

from __future__ import annotations

import argparse
import os

from neda_joint_credit import _atomic_json
from neda_joint_credit_v2 import (
    NEDA_V2_CREDIT_GATE_VERSION,
    materialize,
)
from neda_repro import sha256_file, sha256_json
from neda_v2_decision import NEDA_V2_EVIDENCE_CONTRACT_VERSION
from neda_v4_online_credit import _rollout_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--thought-data", required=True)
    parser.add_argument("--action-data", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--interpolation", type=float, default=0.5)
    parser.add_argument("--evidence-temperature", type=float, default=0.25)
    parser.add_argument("--boundaries-per-thought", type=int, default=4)
    parser.add_argument("--engineering-signal-if-zero", action="store_true")
    parser.add_argument("--require-policy-update-signal", action="store_true")
    args = parser.parse_args()

    rows, manifest = _rollout_rows(args.rollout)
    if (
        manifest.get("environment") != "webshop"
        or manifest.get("rl_method") != "neda"
        or manifest.get("rl_variant") != "neda_v2"
        or manifest.get("credit_evidence_contract")
        != NEDA_V2_EVIDENCE_CONTRACT_VERSION
    ):
        raise ValueError("rollout/NeDA-v2 WebShop protocol mismatch")
    if any(row.get("environment") != "webshop" for row in rows):
        raise ValueError("NeDA-v2 WebShop input contains another environment")

    artifact, thought, action = materialize(
        rows,
        interpolation=args.interpolation,
        evidence_temperature=args.evidence_temperature,
        boundaries_per_thought=args.boundaries_per_thought,
        engineering_signal_if_zero=args.engineering_signal_if_zero,
    )
    artifact.update(
        {
            "environment": "webshop",
            "rollout": os.path.realpath(args.rollout),
            "rollout_sha256": sha256_file(args.rollout),
            "rollout_manifest_sha256": sha256_file(
                args.rollout + ".manifest.json"
            ),
            "behavior_checkpoint_sha256": manifest["model_identity"][
                "identity_sha256"
            ],
        }
    )
    body = dict(artifact)
    body.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = sha256_json(body)
    _atomic_json(artifact, args.out)
    _atomic_json(thought, args.thought_data)
    _atomic_json(action, args.action_data)

    gate = {
        "contract_version": NEDA_V2_CREDIT_GATE_VERSION,
        "status": "PASS",
        "environment": "webshop",
        "method_id": "neda_v2",
        "learner_registered_method": "neda",
        "credit_artifact": os.path.realpath(args.out),
        "credit_artifact_sha256": sha256_file(args.out),
        "thought_data": os.path.realpath(args.thought_data),
        "thought_data_sha256": sha256_file(args.thought_data),
        "action_data": os.path.realpath(args.action_data),
        "action_data_sha256": sha256_file(args.action_data),
        "n_episodes": artifact["n_episodes"],
        "n_turns": artifact["n_turns"],
        "n_thought_records": artifact["n_thought_records"],
        "n_action_records": artifact["n_action_records"],
        "n_action_only_turns": artifact["n_action_only_turns"],
        "n_nonzero_episodes": artifact["n_nonzero_episodes"],
        "scientific_result": artifact["scientific_result"],
        "optimization_signal_status": artifact[
            "optimization_signal_status"
        ],
        "policy_update_eligible": artifact["policy_update_eligible"],
    }
    gate["gate_sha256"] = sha256_json(gate)
    _atomic_json(gate, args.gate)
    print(gate)
    if args.require_policy_update_signal and not gate[
        "policy_update_eligible"
    ]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
