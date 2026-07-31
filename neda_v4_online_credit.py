#!/usr/bin/env python3
"""Materialize deployment NeDA R/H/D/T credit on an on-policy rollout.

The counterfactual benchmark is used only through the frozen deployment heads
inside the sealed NeDA estimator artifact.  Online rollout returns are never
used to refit those heads.  This program emits separate Thought and Action
learner views because the two-stage policy has different recorded prefixes and
commit layouts for AO Thought and AR environment-facing Action.
"""

import argparse
import copy
import json
import math
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from neda_data_contract import validate_decision_trace
from neda_hierarchical_credit import allocate_hierarchy, audit_hierarchy
from neda_hierarchical_credit_builder import (
    ACTION_TOKEN_FEATURES,
    STEP_FEATURES,
    THOUGHT_TOKEN_FEATURES,
    TURN_FEATURES,
    action_token_feature_vector,
    step_feature_vector,
    thought_token_feature_vector,
    turn_feature_vector,
)
from neda_repro import sha256_file, sha256_json
from neda_v4_credit_estimator import ESTIMATOR_CONTRACT_VERSION
from neda_v4_multitrace import validate_action_replay_support


ONLINE_CREDIT_CONTRACT_VERSION = "neda-v4-online-credit-v1"
ONLINE_CREDIT_GATE_VERSION = "neda-v4-online-credit-gate-v1"
EXPECTED_FEATURES = {
    "turn": TURN_FEATURES,
    "step": STEP_FEATURES,
    "thought_token": THOUGHT_TOKEN_FEATURES,
    "action_token": ACTION_TOKEN_FEATURES,
}
ESTIMATOR_ENVIRONMENTS = ("alfworld", "webshop")


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


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} is non-finite".format(label))
    return result


def _predict(head: Mapping[str, Any], features: Sequence[float], level: str) -> float:
    expected = list(EXPECTED_FEATURES[level])
    if list(head.get("feature_names", [])) != expected:
        raise ValueError("{} deployment feature schema drift".format(level))
    features = [_finite(value, "{} feature".format(level)) for value in features]
    if len(features) != len(expected) or int(head.get("n_features", -1)) != len(expected):
        raise ValueError("{} deployment feature width drift".format(level))
    model = head.get("deployment_model", {})
    means = [float(value) for value in model.get("feature_mean", [])]
    scales = [float(value) for value in model.get("feature_scale", [])]
    weights = [float(value) for value in model.get("weights", [])]
    if len(means) != len(features) or len(scales) != len(features):
        raise ValueError("{} deployment transform width drift".format(level))
    if len(weights) != len(features) + 1 or any(value <= 0.0 for value in scales):
        raise ValueError("{} deployment model is malformed".format(level))
    prediction = weights[-1] + sum(
        ((value - mean) / scale) * weight
        for value, mean, scale, weight in zip(features, means, scales, weights[:-1])
    )
    return _finite(prediction, "{} deployment prediction".format(level))


def _project_alfworld_estimator(estimator: Mapping[str, Any]) -> Dict[str, Any]:
    """Authenticate the full two-environment artifact, then select ALF heads.

    The legacy E06/one-step/screening materializer is ALFWorld-only.  Its
    scientific estimator artifact is nevertheless the full sealed eight-head
    object evaluated by Gate O2.  Requiring all eight heads here prevents a
    silently truncated four-head artifact from passing while keeping the
    existing materializer's ``alfworld:<level>`` coordinate contract.
    """

    expected = {
        "{}:{}".format(environment, level)
        for environment in ESTIMATOR_ENVIRONMENTS
        for level in EXPECTED_FEATURES
    }
    heads = estimator.get("heads", {})
    if set(heads) != expected:
        raise ValueError("NeDA estimator environment/head set drift")
    selected = {}
    for level in EXPECTED_FEATURES:
        head = copy.deepcopy(heads["alfworld:" + level])
        if len(str(head.get("head_sha256", ""))) != 64:
            raise ValueError("ALFWorld deployment head lacks a SHA receipt")
        selected["alfworld:" + level] = head
    projected = copy.deepcopy(dict(estimator))
    projected["heads"] = selected
    projected["deployment_head_environment"] = "alfworld"
    return projected


def _verify_estimator(estimator_path: str, o2_gate_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    estimator = _load(estimator_path)
    if (
        not isinstance(estimator, dict)
        or estimator.get("contract_version") != ESTIMATOR_CONTRACT_VERSION
        or estimator.get("status") != "PASS"
        or estimator.get("method") != "neda"
    ):
        raise ValueError("online NeDA requires a sealed PASS NeDA estimator")
    body = dict(estimator)
    stored = str(body.pop("artifact_sha256", ""))
    if len(stored) != 64 or sha256_json(body) != stored:
        raise ValueError("NeDA estimator artifact SHA drift")
    projected = _project_alfworld_estimator(estimator)

    gate = _load(o2_gate_path)
    if (
        not isinstance(gate, dict)
        or gate.get("contract_version") != "neda-v4-o2-fidelity-gate-v1"
        or gate.get("status") != "PASS"
        or int(gate.get("n_fidelity_pass_levels", 0)) < 2
        or gate.get("deletion_pass") is not True
    ):
        raise ValueError("Gate O2 is not a scientific PASS")
    report_path = os.path.realpath(str(gate.get("report_path", "")))
    if not os.path.isfile(report_path) or sha256_file(report_path) != gate.get("report_file_sha256"):
        raise ValueError("Gate O2 report receipt drift")
    report = _load(report_path)
    receipt = report.get("estimator_receipts", {}).get("neda", {})
    if (
        os.path.realpath(str(receipt.get("path", ""))) != os.path.realpath(estimator_path)
        or receipt.get("sha256") != sha256_file(estimator_path)
        or receipt.get("artifact_sha256") != estimator.get("artifact_sha256")
    ):
        raise ValueError("Gate O2 did not evaluate this NeDA estimator artifact")
    return projected, gate


def _rollout_rows(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = _load(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("online rollout must be a non-empty JSON list")
    manifest_path = path + ".manifest.json"
    manifest = _load(manifest_path)
    if manifest.get("data_sha256") != sha256_file(path):
        raise ValueError("online rollout manifest SHA drift")
    if manifest.get("online_credit_context_contract") != "neda-online-credit-state-v1":
        raise ValueError("online rollout lacks the frozen credit-state contract")
    for row in rows:
        required = (
            "sample_id", "game_id", "group_id", "episode_id", "turn_id",
            "episode_horizon", "episode_return", "group_advantage",
            "turn_reward", "state_before", "decision_traces",
        )
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError("online rollout row is missing {}".format(missing))
        if row.get("raw_action") != row.get("executed_action"):
            raise ValueError("NeDA online rollout requires raw Action == executed Action")
        state = row["state_before"]
        if state.get("contract_version") != "neda-online-credit-state-v1":
            raise ValueError("online credit-state contract drift")
        thought = row["decision_traces"].get("thought")
        action = row["decision_traces"].get("action")
        if not isinstance(thought, dict):
            raise ValueError("online rollout is missing its Thought trace")
        # An Action-only decision is a real policy outcome, not a malformed
        # sample.  Its zero-length interface Thought remains authenticated by
        # the native replay state with an all-false optimization mask.  We do
        # not invent placeholder Thought tokens or assign them credit.
        validate_decision_trace(
            thought, require_logprobs=True, require_sampling=True,
            exact_replay=False,
        )
        if not isinstance(action, dict) or not action.get("response_ids"):
            raise ValueError("online rollout has an empty action trace")
        validate_action_replay_support(action)
    return rows, manifest


def _episodes(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["episode_id"])].append(row)
    episodes = []
    for episode_id, selected in sorted(grouped.items()):
        selected = sorted(selected, key=lambda row: int(row["turn_id"]))
        horizon = int(selected[0]["episode_horizon"])
        if len(selected) != horizon or [int(row["turn_id"]) for row in selected] != list(range(horizon)):
            raise ValueError("episode turn sequence/horizon drift: {}".format(episode_id))
        invariant = (
            "game_id", "group_id", "rollout_id", "episode_return",
            "group_advantage", "model_identity_sha256",
        )
        for key in invariant:
            if len({str(row[key]) for row in selected}) != 1:
                raise ValueError("episode invariant drift for {}".format(key))
        episodes.append(
            {
                "game_id": str(selected[0]["game_id"]),
                "group_id": str(selected[0]["group_id"]),
                "episode_id": episode_id,
                "rollout_id": int(selected[0]["rollout_id"]),
                "horizon": horizon,
                "return": _finite(selected[0]["episode_return"], "episode return"),
                "group_advantage": _finite(
                    selected[0]["group_advantage"], "group advantage"
                ),
                "turns": [dict(row) for row in selected],
            }
        )
    # Recompute the GRPO group normalization instead of trusting a convenient
    # scalar copied into every turn.
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_group[episode["group_id"]].append(episode)
    for group_id, selected in by_group.items():
        if len({episode["game_id"] for episode in selected}) != 1:
            raise ValueError("one rollout group crossed game IDs")
        returns = [episode["return"] for episode in selected]
        mean = statistics.mean(returns)
        scale = statistics.pstdev(returns) if len(returns) > 1 else 0.0
        for episode in selected:
            expected = (episode["return"] - mean) / (scale + 1e-6) if scale > 0 else 0.0
            if abs(expected - episode["group_advantage"]) > 1e-8:
                raise ValueError("stored group advantage drift in {}".format(group_id))
    return episodes


def _positions(trace: Mapping[str, Any], step_id: int) -> List[int]:
    result = [
        index for index, value in enumerate(trace["step_map"])
        if int(value) == int(step_id)
    ]
    if not result:
        raise ValueError("recorded Thought step has no committed tokens")
    return result


def materialize(
    rows: Sequence[Mapping[str, Any]],
    estimator: Mapping[str, Any],
    eta_h: float,
    eta_d: float,
    eta_t: float,
    outer_prior: str,
    gamma: float,
    mass_constant: float,
    registered_method: str = "neda",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not registered_method or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in registered_method
    ):
        raise ValueError("registered method identifier is malformed")
    episodes = _episodes(rows)
    heads = {
        level: estimator["heads"]["alfworld:" + level]
        for level in EXPECTED_FEATURES
    }
    thought_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    episode_artifacts = []
    maximum = {"turn": 0.0, "step": 0.0, "token": 0.0}
    n_nonzero = 0
    for episode in episodes:
        turn_scores: List[float] = []
        step_scores: List[List[float]] = []
        token_scores: List[List[List[float]]] = []
        metadata: List[List[Dict[str, Any]]] = []
        for turn in episode["turns"]:
            turn_scores.append(
                _predict(heads["turn"], turn_feature_vector(episode, turn), "turn")
            )
            trace = turn["decision_traces"]["thought"]
            local_steps: List[float] = []
            local_tokens: List[List[float]] = []
            local_metadata: List[Dict[str, Any]] = []
            for step_id in sorted(set(int(value) for value in trace["step_map"])):
                positions = _positions(trace, step_id)
                local_steps.append(
                    _predict(
                        heads["step"],
                        step_feature_vector(episode, turn, step_id, positions),
                        "step",
                    )
                )
                local_tokens.append(
                    [
                        _predict(
                            heads["thought_token"],
                            thought_token_feature_vector(episode, turn, position),
                            "thought_token",
                        )
                        for position in positions
                    ]
                )
                local_metadata.append(
                    {"kind": "thought", "recorded_step_id": step_id, "positions": positions}
                )
            action = turn["decision_traces"]["action"]
            action_values = [
                _predict(
                    heads["action_token"],
                    action_token_feature_vector(episode, turn, position),
                    "action_token",
                )
                for position in range(len(action["response_ids"]))
            ]
            local_steps.append(sum(action_values) / len(action_values))
            local_tokens.append(action_values)
            local_metadata.append(
                {
                    "kind": "action",
                    "recorded_step_id": None,
                    "positions": list(range(len(action_values))),
                    "parent_score": "mean-action-token-deployment-prediction",
                }
            )
            step_scores.append(local_steps)
            token_scores.append(local_tokens)
            metadata.append(local_metadata)

        hierarchy = allocate_hierarchy(
            episode["group_advantage"], mass_constant, turn_scores,
            step_scores, token_scores, eta_h, eta_d, eta_t,
            outer_prior=outer_prior, gamma=gamma,
        )
        audit = audit_hierarchy(hierarchy)
        maximum["turn"] = max(maximum["turn"], audit["turn_mass_error"])
        maximum["step"] = max(maximum["step"], audit["max_step_parent_error"])
        maximum["token"] = max(maximum["token"], audit["max_token_parent_error"])
        if abs(hierarchy["episode_mass"]) > 1e-12:
            n_nonzero += 1

        for turn_credit, source_turn, step_meta in zip(
            hierarchy["turns"], episode["turns"], metadata
        ):
            thought_adv = [None] * len(
                source_turn["decision_traces"]["thought"]["response_ids"]
            )
            action_adv = [None] * len(
                source_turn["decision_traces"]["action"]["response_ids"]
            )
            for step_credit, meta in zip(turn_credit["steps"], step_meta):
                step_credit.update(meta)
                target = thought_adv if meta["kind"] == "thought" else action_adv
                for token_credit, position in zip(step_credit["tokens"], meta["positions"]):
                    token_credit["recorded_token_position"] = int(position)
                    if target[position] is not None:
                        raise ValueError("one recorded token received duplicate credit")
                    target[position] = _finite(token_credit["credit"], "token credit")
            if any(value is None for value in thought_adv + action_adv):
                raise ValueError("hierarchy did not cover every recorded policy token")
            turn_credit["turn_id"] = int(source_turn["turn_id"])

            for kind, adv_map, destination in (
                ("thought", thought_adv, thought_rows),
                ("action", action_adv, action_rows),
            ):
                if not adv_map:
                    # Action-only decisions have no Thought policy coordinate.
                    # The Action learner record below still carries the full
                    # hierarchy allocation; no placeholder token is created.
                    if kind == "thought":
                        continue
                    raise ValueError("online Action trace unexpectedly has no tokens")
                learner = copy.deepcopy(source_turn)
                learner["sample_id"] = "{}::{}".format(source_turn["sample_id"], kind)
                learner["credit_trace_kind"] = kind
                learner["reward"] = float(turn_credit["credit"])
                learner["adv_map"] = [float(value) for value in adv_map]
                learner["neda_credit_contract"] = ONLINE_CREDIT_CONTRACT_VERSION
                learner["registered_method"] = registered_method
                learner["neda_episode_credit_sha256"] = hierarchy["credit_sha256"]
                destination.append(learner)

        episode_artifacts.append(
            {
                "game_id": episode["game_id"],
                "group_id": episode["group_id"],
                "episode_id": episode["episode_id"],
                "rollout_id": episode["rollout_id"],
                "return": episode["return"],
                "group_advantage": episode["group_advantage"],
                "hierarchy": hierarchy,
            }
        )

    result = {
        "contract_version": ONLINE_CREDIT_CONTRACT_VERSION,
        "status": "PASS",
        "environment": "alfworld",
        "registered_method": registered_method,
        "estimator_artifact_sha256": estimator["artifact_sha256"],
        "design": {
            "outer_prior": outer_prior,
            "gamma": float(gamma),
            "mass_constant": float(mass_constant),
            "eta": {"H": float(eta_h), "D": float(eta_d), "T": float(eta_t)},
            "thought_policy_coordinate": "recorded AO diffusion commits",
            "action_policy_coordinate": "recorded AR environment-Action tokens",
            "action_parent": "joint Action is one D-level child of its environment turn",
        },
        "n_episodes": len(episode_artifacts),
        "n_turns": sum(episode["horizon"] for episode in episodes),
        "n_thought_learner_records": len(thought_rows),
        "n_action_learner_records": len(action_rows),
        "n_action_only_turns": len(action_rows) - len(thought_rows),
        "n_nonzero_episode_masses": n_nonzero,
        "maximum_mass_error": maximum,
        "episodes": episode_artifacts,
    }
    result["artifact_sha256"] = sha256_json(result)
    return result, thought_rows, action_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--estimator", required=True)
    parser.add_argument("--o2-gate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--thought-data", required=True)
    parser.add_argument("--action-data", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--eta-h", type=float, default=0.5)
    parser.add_argument("--eta-d", type=float, default=0.5)
    parser.add_argument("--eta-t", type=float, default=0.5)
    parser.add_argument("--outer-prior", choices=("uniform", "temporal"), default="uniform")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--mass-constant", type=float, default=1.0)
    parser.add_argument("--registered-method", default="neda")
    args = parser.parse_args()
    if any(not 0.0 <= value <= 1.0 for value in (args.eta_h, args.eta_d, args.eta_t)):
        raise ValueError("eta values must lie in [0,1]")
    if not 0.0 < args.gamma <= 1.0 or args.mass_constant <= 0.0:
        raise ValueError("gamma/mass_constant are outside the registered domain")

    estimator, o2_gate = _verify_estimator(args.estimator, args.o2_gate)
    rows, rollout_manifest = _rollout_rows(args.rollout)
    artifact, thought_rows, action_rows = materialize(
        rows, estimator, args.eta_h, args.eta_d, args.eta_t,
        args.outer_prior, args.gamma, args.mass_constant, args.registered_method,
    )
    artifact["rollout_path"] = os.path.realpath(args.rollout)
    artifact["rollout_sha256"] = sha256_file(args.rollout)
    artifact["rollout_manifest_sha256"] = sha256_file(args.rollout + ".manifest.json")
    artifact["behavior_checkpoint_sha256"] = rollout_manifest["model_identity"]["identity_sha256"]
    artifact["o2_gate_sha256"] = o2_gate["gate_sha256"]
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = sha256_json(artifact)
    _atomic_json(artifact, args.out)
    _atomic_json(thought_rows, args.thought_data)
    _atomic_json(action_rows, args.action_data)
    gate = {
        "contract_version": ONLINE_CREDIT_GATE_VERSION,
        "status": "PASS",
        "credit_artifact": os.path.realpath(args.out),
        "credit_artifact_sha256": sha256_file(args.out),
        "scientific_artifact_sha256": artifact["artifact_sha256"],
        "thought_data": os.path.realpath(args.thought_data),
        "thought_data_sha256": sha256_file(args.thought_data),
        "action_data": os.path.realpath(args.action_data),
        "action_data_sha256": sha256_file(args.action_data),
        "n_episodes": artifact["n_episodes"],
        "n_turns": artifact["n_turns"],
        "n_nonzero_episode_masses": artifact["n_nonzero_episode_masses"],
        "maximum_mass_error": artifact["maximum_mass_error"],
    }
    gate["gate_sha256"] = sha256_json(gate)
    _atomic_json(gate, args.gate)
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
