#!/usr/bin/env python3
"""Mass-matched ALFWorld credit for MAPG, DCoLT, and current NeDA.

All three methods receive the same episode group advantage and the same
environment-turn mass A/H.  MAPG and DCoLT distribute that mass uniformly over
the realized Thought commitment steps.  NeDA conservatively interpolates the
uniform allocation with normalized evidence from how StepMerge segments change
the teacher-forced probability of the Action actually sent to ALFWorld.

No value network or purported ground-truth per-step credit is used.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from neda_joint_policy import JOINT_METHODS, method_position_policy
from neda_repro import sha256_file, sha256_json
from neda_v4_multitrace import validate_action_replay_support
from neda_v4_online_credit import _episodes, _rollout_rows


JOINT_CREDIT_CONTRACT_VERSION = "neda-joint-credit-v1"
JOINT_CREDIT_GATE_VERSION = "neda-joint-credit-gate-v1"


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


def _softplus(value: float) -> float:
    value = max(-40.0, min(40.0, float(value)))
    return math.log1p(math.exp(value))


def _rounds(trace: Mapping[str, Any]) -> List[int]:
    values = sorted(set(int(value) for value in trace["step_map"]))
    if not values or values[0] < 0:
        raise ValueError("Thought policy trace contains an uncommitted token")
    return values


def _uniform_weights(rounds: Sequence[int]) -> Dict[int, float]:
    mass = 1.0 / len(rounds)
    return {int(value): mass for value in rounds}


def _select_boundaries(rounds: Sequence[int], count: int) -> List[int]:
    values = sorted(set(int(value) for value in rounds))
    count = min(max(1, int(count)), len(values))
    if count == 1:
        return [values[-1]]
    indices = {
        round(index * (len(values) - 1) / (count - 1))
        for index in range(count)
    }
    return [values[index] for index in sorted(indices)]


def _neda_weights(
    trace: Mapping[str, Any],
    interpolation: float,
    evidence_temperature: float,
    boundaries_per_thought: int,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    rounds = _rounds(trace)
    evidence = trace.get("action_evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("contract_version") != "neda-action-evidence-v1"
    ):
        raise ValueError("NeDA rollout lacks Action-mediated evidence")
    raw: Dict[int, float] = {}
    segments = []
    for segment in evidence.get("segments", []):
        members = [int(value) for value in segment["member_rounds"]]
        delta = float(segment["action_logprob_delta"])
        segment_weight = _softplus(delta / float(evidence_temperature))
        if not members:
            raise ValueError("Action-evidence segment has no denoising rounds")
        boundary = int(segment["right_round_inclusive"])
        if boundary in raw:
            raise ValueError("Action-evidence boundary received duplicate allocation")
        raw[boundary] = segment_weight
        segments.append(
            {
                "member_rounds": members,
                "action_logprob_delta": delta,
                "positive_evidence": segment_weight,
            }
        )
    expected = _select_boundaries(rounds, boundaries_per_thought)
    if sorted(raw) != expected:
        raise ValueError(
            "Action-evidence and configured StepMerge boundaries disagree"
        )
    denominator = sum(raw.values())
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("NeDA Action evidence cannot be normalized")
    evidence_weights = {
        round_id: value / denominator for round_id, value in raw.items()
    }
    uniform = _uniform_weights(expected)
    result = {
        round_id: (
            (1.0 - float(interpolation)) * uniform[round_id]
            + float(interpolation) * evidence_weights[round_id]
        )
        for round_id in expected
    }
    error = abs(sum(result.values()) - 1.0)
    if error > 1e-10 or any(value < 0.0 for value in result.values()):
        raise ValueError("NeDA denoising-step allocation is not a simplex")
    return result, {
        "segments": segments,
        "uniform_weights": {str(key): value for key, value in uniform.items()},
        "evidence_weights": {
            str(key): value for key, value in evidence_weights.items()
        },
        "interpolation": float(interpolation),
        "mass_error": error,
    }


def _copy_record(
    turn: Mapping[str, Any],
    kind: str,
    method: str,
    turn_mass: float,
    step_weights: Mapping[int, float] | None,
) -> Dict[str, Any]:
    result = copy.deepcopy(dict(turn))
    trace = result["decision_traces"][kind]
    if kind == "action":
        validate_action_replay_support(trace)
        adv_map = [float(turn_mass)] * len(trace["response_ids"])
        step_credit = {}
    else:
        if not step_weights:
            raise ValueError("Thought record requires denoising-step weights")
        step_credit = {
            int(round_id): float(turn_mass) * float(weight)
            for round_id, weight in step_weights.items()
        }
        adv_map = [
            step_credit.get(int(round_id), 0.0)
            for round_id in trace["step_map"]
        ]
    result["sample_id"] = "{}::{}::{}".format(
        turn["sample_id"], method, kind
    )
    result["credit_trace_kind"] = kind
    result["registered_method"] = method
    result["method_credit_contract"] = JOINT_CREDIT_CONTRACT_VERSION
    result["reward"] = float(turn_mass)
    result["adv_map"] = adv_map
    result["step_credit_by_round"] = {
        str(key): value for key, value in step_credit.items()
    }
    result["position_policy"] = method_position_policy(method)
    result["position_objective"] = {
        "mapg": "separately clipped token and Plackett-Luce position ratios",
        "dcolt": "joint token plus learned-UPM position ratio",
        "neda": "MAPG joint coordinate with Action-mediated step coefficients",
    }[method]
    return result


def materialize(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    *,
    interpolation: float,
    evidence_temperature: float,
    boundaries_per_thought: int,
    engineering_signal_if_zero: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    method = str(method).lower()
    if method not in JOINT_METHODS:
        raise ValueError("unknown joint method")
    if not 0.0 <= float(interpolation) <= 1.0:
        raise ValueError("NeDA interpolation must lie in [0,1]")
    if float(evidence_temperature) <= 0.0:
        raise ValueError("evidence temperature must be positive")
    if int(boundaries_per_thought) <= 0:
        raise ValueError("boundaries-per-thought must be positive")
    episodes = _episodes(rows)
    override = bool(engineering_signal_if_zero) and not any(
        abs(float(episode["group_advantage"])) > 1e-12 for episode in episodes
    )
    thought_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    allocation_receipts = []
    nonzero = 0
    max_mass_error = 0.0
    n_turns = 0
    n_action_only_turns = 0
    for episode_index, episode in enumerate(episodes):
        advantage = float(episode["group_advantage"])
        if override:
            advantage = 1.0 if episode_index % 2 == 0 else -1.0
        if abs(advantage) > 1e-12:
            nonzero += 1
        horizon = int(episode["horizon"])
        turn_mass = advantage / float(horizon)
        for turn in episode["turns"]:
            n_turns += 1
            trace = turn["decision_traces"]["thought"]
            has_thought = bool(trace.get("response_ids"))
            if has_thought:
                if method == "neda":
                    weights, diagnostic = _neda_weights(
                        trace,
                        interpolation,
                        evidence_temperature,
                        boundaries_per_thought,
                    )
                else:
                    weights = _uniform_weights(
                        _select_boundaries(
                            _rounds(trace), boundaries_per_thought
                        )
                    )
                    diagnostic = {
                        "uniform_weights": {
                            str(key): value for key, value in weights.items()
                        },
                        "interpolation": 0.0,
                        "mass_error": abs(sum(weights.values()) - 1.0),
                    }
                step_mass = sum(
                    turn_mass * value for value in weights.values()
                )
                mass_error = abs(step_mass - turn_mass)
                max_mass_error = max(max_mass_error, mass_error)
                if mass_error > 1e-9:
                    raise ValueError(
                        "within-turn denoising credit is not mass conserving"
                    )
                thought_rows.append(
                    _copy_record(
                        turn, "thought", method, turn_mass, weights
                    )
                )
            else:
                if trace.get("step_map"):
                    raise ValueError(
                        "Action-only Thought trace contains committed rounds"
                    )
                n_action_only_turns += 1
                weights = {}
                diagnostic = {
                    "allocation_status": "ACTION_ONLY",
                    "policy": (
                        "no synthetic Thought tokens; optimize the recorded "
                        "Action coordinate only"
                    ),
                    "uniform_weights": {},
                    "evidence_weights": {},
                    "interpolation": (
                        float(interpolation) if method == "neda" else 0.0
                    ),
                    "mass_error": 0.0,
                }
            action_rows.append(
                _copy_record(turn, "action", method, turn_mass, None)
            )
            allocation_receipts.append(
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
        "contract_version": JOINT_CREDIT_CONTRACT_VERSION,
        "status": "PASS",
        "environment": "alfworld",
        "method_id": method,
        "position_policy": method_position_policy(method),
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
                "uniform" if method != "neda"
                else "conservative uniform/Action-evidence interpolation"
            ),
            "action_only_policy": (
                "route the turn update only through recorded Action tokens; "
                "never synthesize Thought commitments"
            ),
            "maximum_turn_mass_error": max_mass_error,
        },
        "neda_interpolation": float(interpolation) if method == "neda" else None,
        "boundaries_per_thought": int(boundaries_per_thought),
        "evidence_temperature": (
            float(evidence_temperature) if method == "neda" else None
        ),
        "allocations": allocation_receipts,
    }
    body = dict(artifact)
    body.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = sha256_json(body)
    return artifact, thought_rows, action_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=JOINT_METHODS, required=True)
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--thought-data", required=True)
    parser.add_argument("--action-data", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--interpolation", type=float, default=0.5)
    parser.add_argument("--evidence-temperature", type=float, default=0.25)
    parser.add_argument("--boundaries-per-thought", type=int, default=8)
    parser.add_argument("--engineering-signal-if-zero", action="store_true")
    parser.add_argument(
        "--require-policy-update-signal",
        action="store_true",
        help=(
            "Write the authenticated credit/gate artifacts, then exit 20 if "
            "every scientific GRPO group has zero return variance."
        ),
    )
    args = parser.parse_args()
    rows, manifest = _rollout_rows(args.rollout)
    if manifest.get("rl_method") != args.method:
        raise ValueError("rollout/method mismatch")
    artifact, thought, action = materialize(
        rows,
        args.method,
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
            "behavior_dcolt_head_sha256": manifest.get("dcolt_head_sha256"),
        }
    )
    body = dict(artifact)
    body.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = sha256_json(body)
    _atomic_json(artifact, args.out)
    _atomic_json(thought, args.thought_data)
    _atomic_json(action, args.action_data)
    gate = {
        "contract_version": JOINT_CREDIT_GATE_VERSION,
        "status": "PASS",
        "method_id": args.method,
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
            "[joint-credit] no within-group return variation; refusing an "
            "uninformative formal policy update",
            flush=True,
        )
        raise SystemExit(20)


if __name__ == "__main__":
    main()
