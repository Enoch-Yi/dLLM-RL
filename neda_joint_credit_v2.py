#!/usr/bin/env python3
"""Mass-conserving NeDA-v2 denoising commitment credit.

Every episode keeps the same group-relative advantage and every environment
turn keeps the same ``A_i / H_i`` mass as the matched NeDA-v1/MAPG runs.
Inside a Thought, NeDA-v2 applies a temperature-softmax to changes in the
executed-vs-other-legal Action log-odds margin, then conservatively mixes the
result with uniform StepMerge credit.  There is no critic and no claim that
these coefficients are oracle causal effects.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from neda_joint_credit import (
    _atomic_json,
    _copy_record,
    _rounds,
    _select_boundaries,
    _uniform_weights,
)
from neda_joint_policy import method_position_policy
from neda_repro import sha256_file, sha256_json
from neda_v2_decision import NEDA_V2_EVIDENCE_CONTRACT_VERSION
from neda_v4_online_credit import _episodes, _rollout_rows


NEDA_V2_CREDIT_CONTRACT_VERSION = "neda-v2-joint-credit-v1"
NEDA_V2_CREDIT_GATE_VERSION = "neda-v2-joint-credit-gate-v1"


def _softmax_by_round(
    values: Mapping[int, float], temperature: float
) -> Dict[int, float]:
    if not values:
        raise ValueError("NeDA-v2 evidence cannot be empty")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("NeDA-v2 evidence temperature must be positive")
    scaled = {
        int(key): float(value) / temperature for key, value in values.items()
    }
    if any(not math.isfinite(value) for value in scaled.values()):
        raise ValueError("NeDA-v2 evidence delta is non-finite")
    maximum = max(scaled.values())
    exponentials = {
        key: math.exp(value - maximum) for key, value in scaled.items()
    }
    denominator = math.fsum(exponentials.values())
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("NeDA-v2 evidence softmax cannot be normalized")
    return {key: value / denominator for key, value in exponentials.items()}


def neda_v2_weights(
    trace: Mapping[str, Any],
    interpolation: float,
    evidence_temperature: float,
    boundaries_per_thought: int,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    rounds = _rounds(trace)
    evidence = trace.get("action_evidence_v2")
    if (
        not isinstance(evidence, dict)
        or evidence.get("contract_version")
        != NEDA_V2_EVIDENCE_CONTRACT_VERSION
    ):
        raise ValueError("NeDA-v2 rollout lacks legal-Action margin evidence")
    raw: Dict[int, float] = {}
    segments = []
    for segment in evidence.get("segments", []):
        members = [int(value) for value in segment["member_rounds"]]
        if not members:
            raise ValueError("NeDA-v2 evidence segment has no denoising rounds")
        boundary = int(segment["right_round_inclusive"])
        delta = float(segment["legal_action_margin_delta"])
        if boundary in raw:
            raise ValueError("NeDA-v2 boundary received duplicate evidence")
        raw[boundary] = delta
        segments.append(
            {
                "member_rounds": members,
                "legal_action_margin_delta": delta,
            }
        )
    expected = _select_boundaries(rounds, boundaries_per_thought)
    if sorted(raw) != expected:
        raise ValueError(
            "NeDA-v2 evidence and configured StepMerge boundaries disagree"
        )
    evidence_weights = _softmax_by_round(raw, evidence_temperature)
    uniform = _uniform_weights(expected)
    result = {
        round_id: (
            (1.0 - float(interpolation)) * uniform[round_id]
            + float(interpolation) * evidence_weights[round_id]
        )
        for round_id in expected
    }
    error = abs(math.fsum(result.values()) - 1.0)
    if error > 1.0e-10 or any(value < 0.0 for value in result.values()):
        raise ValueError("NeDA-v2 denoising allocation is not a simplex")
    return result, {
        "allocation_status": evidence.get("allocation_status"),
        "segments": segments,
        "uniform_weights": {str(key): value for key, value in uniform.items()},
        "evidence_weights": {
            str(key): value for key, value in evidence_weights.items()
        },
        "interpolation": float(interpolation),
        "evidence_temperature": float(evidence_temperature),
        "mass_error": error,
        "evidence_sha256": evidence.get("evidence_sha256"),
    }


def _v2_record(
    turn: Mapping[str, Any],
    kind: str,
    turn_mass: float,
    step_weights: Mapping[int, float] | None,
) -> Dict[str, Any]:
    result = _copy_record(turn, kind, "neda", turn_mass, step_weights)
    result["sample_id"] = "{}::neda_v2::{}".format(turn["sample_id"], kind)
    result["registered_method"] = "neda"
    result["method_variant"] = "neda_v2"
    result["method_credit_contract"] = NEDA_V2_CREDIT_CONTRACT_VERSION
    result["position_objective"] = (
        "MAPG token/position policy coordinate with legal-Action "
        "contrastive StepMerge coefficients"
    )
    return result


def materialize(
    rows: Sequence[Mapping[str, Any]],
    *,
    interpolation: float,
    evidence_temperature: float,
    boundaries_per_thought: int,
    engineering_signal_if_zero: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not 0.0 <= float(interpolation) <= 1.0:
        raise ValueError("NeDA-v2 interpolation must lie in [0,1]")
    if int(boundaries_per_thought) <= 0:
        raise ValueError("boundaries-per-thought must be positive")
    episodes = _episodes(rows)
    override = bool(engineering_signal_if_zero) and not any(
        abs(float(episode["group_advantage"])) > 1.0e-12
        for episode in episodes
    )
    thought_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    allocations = []
    nonzero = 0
    n_turns = 0
    n_action_only_turns = 0
    max_mass_error = 0.0

    for episode_index, episode in enumerate(episodes):
        advantage = float(episode["group_advantage"])
        if override:
            advantage = 1.0 if episode_index % 2 == 0 else -1.0
        if abs(advantage) > 1.0e-12:
            nonzero += 1
        horizon = int(episode["horizon"])
        turn_mass = advantage / float(horizon)
        for turn in episode["turns"]:
            n_turns += 1
            trace = turn["decision_traces"]["thought"]
            has_thought = bool(trace.get("response_ids"))
            if has_thought:
                weights, diagnostic = neda_v2_weights(
                    trace,
                    interpolation,
                    evidence_temperature,
                    boundaries_per_thought,
                )
                allocated = math.fsum(
                    turn_mass * value for value in weights.values()
                )
                mass_error = abs(allocated - turn_mass)
                max_mass_error = max(max_mass_error, mass_error)
                if mass_error > 1.0e-9:
                    raise ValueError("NeDA-v2 within-turn mass is not conserved")
                thought_rows.append(
                    _v2_record(turn, "thought", turn_mass, weights)
                )
            else:
                if trace.get("step_map"):
                    raise ValueError(
                        "Action-only Thought trace contains committed rounds"
                    )
                evidence = trace.get("action_evidence_v2", {})
                if evidence.get("allocation_status") != "ACTION_ONLY":
                    raise ValueError("Action-only turn lacks NeDA-v2 evidence")
                n_action_only_turns += 1
                weights = {}
                diagnostic = {
                    "allocation_status": "ACTION_ONLY",
                    "policy": "optimize only the recorded Action coordinate",
                    "uniform_weights": {},
                    "evidence_weights": {},
                    "interpolation": float(interpolation),
                    "evidence_temperature": float(evidence_temperature),
                    "mass_error": 0.0,
                }
            action_rows.append(_v2_record(turn, "action", turn_mass, None))
            allocations.append(
                {
                    "episode_id": episode["episode_id"],
                    "turn_id": int(turn["turn_id"]),
                    "episode_advantage": advantage,
                    "episode_horizon": horizon,
                    "turn_mass": turn_mass,
                    "decision_case": (
                        "THOUGHT_AND_ACTION" if has_thought else "ACTION_ONLY"
                    ),
                    "step_credit_by_round": {
                        str(key): turn_mass * value
                        for key, value in weights.items()
                    },
                    "diagnostic": diagnostic,
                }
            )

    artifact = {
        "contract_version": NEDA_V2_CREDIT_CONTRACT_VERSION,
        "status": "PASS",
        "environment": "alfworld",
        "method_id": "neda_v2",
        "learner_registered_method": "neda",
        "position_policy": method_position_policy("neda"),
        "n_episodes": len(episodes),
        "n_turns": n_turns,
        "n_thought_records": len(thought_rows),
        "n_action_records": len(action_rows),
        "n_action_only_turns": n_action_only_turns,
        "n_nonzero_episodes": nonzero,
        "engineering_advantage_override": override,
        "scientific_result": not override,
        "optimization_signal_status": (
            "ENGINEERING_OVERRIDE"
            if override
            else ("PASS" if nonzero > 0 else "ZERO_GROUP_VARIANCE")
        ),
        "policy_update_eligible": bool(override or nonzero > 0),
        "mass_conservation": {
            "episode_to_turn": "A_i / H_i",
            "turn_to_denoising": (
                "uniform/contrastive legal-Action margin interpolation"
            ),
            "action_only_policy": (
                "route the turn update only through recorded Action tokens"
            ),
            "maximum_turn_mass_error": max_mass_error,
        },
        "neda_interpolation": float(interpolation),
        "boundaries_per_thought": int(boundaries_per_thought),
        "evidence_temperature": float(evidence_temperature),
        "evidence_contract": NEDA_V2_EVIDENCE_CONTRACT_VERSION,
        "allocations": allocations,
    }
    artifact["artifact_sha256"] = sha256_json(artifact)
    return artifact, thought_rows, action_rows


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
        manifest.get("rl_method") != "neda"
        or manifest.get("rl_variant") != "neda_v2"
        or manifest.get("credit_evidence_contract")
        != NEDA_V2_EVIDENCE_CONTRACT_VERSION
    ):
        raise ValueError("rollout/NeDA-v2 protocol mismatch")
    artifact, thought, action = materialize(
        rows,
        interpolation=args.interpolation,
        evidence_temperature=args.evidence_temperature,
        boundaries_per_thought=args.boundaries_per_thought,
        engineering_signal_if_zero=args.engineering_signal_if_zero,
    )
    artifact.update(
        {
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
        "optimization_signal_status": artifact["optimization_signal_status"],
        "policy_update_eligible": artifact["policy_update_eligible"],
    }
    gate["gate_sha256"] = sha256_json(gate)
    _atomic_json(gate, args.gate)
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    if args.require_policy_update_signal and not gate["policy_update_eligible"]:
        print(
            "[neda-v2-credit] no within-group return variation; refusing an "
            "uninformative formal policy update",
            flush=True,
        )
        raise SystemExit(20)


if __name__ == "__main__":
    main()
