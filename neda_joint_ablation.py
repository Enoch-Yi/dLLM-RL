#!/usr/bin/env python3
"""Prepare registered NeDA component-ablation learner artifacts.

The current formal learner is deliberately left unchanged.  This module
transforms already authenticated NeDA credit artifacts into narrowly defined
ablation inputs:

* ``uniform`` and ``evidence_only`` keep the joint token/position/Action
  objective and deterministically reweight an authenticated NeDA artifact
  from its recorded uniform/evidence simplex diagnostics;
* ``no_position`` keeps Thought-token and Action-token credit but routes the
  learner through its exact token-only ratio path;
* ``token_only`` additionally zeros every Action coefficient;
* ``no_horizon`` copies the episode advantage to every turn by multiplying
  the otherwise mass-conserving turn credit by the episode horizon;
* ``random_path`` is an expected-failure replay diagnostic, never a training
  run.  It changes one recorded unmask-position decision while leaving the
  behavior likelihood untouched so the exact-path gate must reject it.

No variant changes the rollout, environment return, task schedule, or behavior
checkpoint.  Trainable variants require their own on-policy refresh after the
first update.
"""

import argparse
import copy
import json
import math
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from neda_repro import sha256_file, sha256_json


CONTRACT_VERSION = "neda-joint-ablation-materialization-v1"
GATE_VERSION = "neda-joint-ablation-gate-v1"
VARIANTS = (
    "uniform",
    "evidence_only",
    "no_position",
    "token_only",
    "no_horizon",
    "random_path",
)
GENERIC_METHOD = {
    "no_position": "neda_no_position",
    "token_only": "neda_token_only",
}


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


def _validate_source(
    artifact: Mapping[str, Any],
    thought: Sequence[Mapping[str, Any]],
    action: Sequence[Mapping[str, Any]],
) -> None:
    source_contract = artifact.get("contract_version")
    expected_method = {
        "neda-joint-credit-v1": "neda",
        "neda-v2-joint-credit-v1": "neda_v2",
    }.get(source_contract)
    if expected_method is None:
        raise ValueError("ablation source credit contract drift")
    if (
        artifact.get("status") != "PASS"
        or artifact.get("method_id") != expected_method
    ):
        raise ValueError("ablation source must be a PASS NeDA credit artifact")
    if not thought or not action:
        raise ValueError("ablation source learner artifacts are empty")
    for kind, records in (("thought", thought), ("action", action)):
        for record in records:
            if record.get("credit_trace_kind") != kind:
                raise ValueError("ablation source kind drift")
            if record.get("registered_method") != "neda":
                raise ValueError("ablation source method drift")
            if source_contract == "neda-v2-joint-credit-v1" and record.get(
                "method_variant"
            ) != "neda_v2":
                raise ValueError("ablation source lost NeDA-v2 provenance")
            trace = record.get("decision_traces", {}).get(kind)
            if not isinstance(trace, dict) or not trace.get("response_ids"):
                raise ValueError("ablation source has an empty trace")
            adv_map = record.get("adv_map")
            if not isinstance(adv_map, list) or len(adv_map) != len(
                trace["response_ids"]
            ):
                raise ValueError("ablation source adv-map alignment drift")
            if any(not math.isfinite(float(value)) for value in adv_map):
                raise ValueError("ablation source has non-finite credit")


def _rename_record(record: Dict[str, Any], method: str, kind: str) -> None:
    sample_id = str(record["sample_id"])
    pieces = sample_id.rsplit("::", 2)
    prefix = pieces[0] if len(pieces) == 3 else sample_id
    record["sample_id"] = "{}::{}::{}".format(prefix, method, kind)
    record["registered_method"] = method
    record["position_policy"] = "not_optimized"
    record["position_objective"] = (
        "exact recorded token ratios; unmask-position ratio disabled"
    )


def _scale_record(record: Dict[str, Any], scale: float) -> None:
    record["reward"] = _finite(record.get("reward", 0.0), "record reward") * scale
    record["adv_map"] = [
        _finite(value, "record credit") * scale
        for value in record["adv_map"]
    ]
    record["step_credit_by_round"] = {
        str(key): _finite(value, "step credit") * scale
        for key, value in record.get("step_credit_by_round", {}).items()
    }


def _reweight_neda(
    result: Dict[str, Any],
    thought: Sequence[Dict[str, Any]],
    action: Sequence[Dict[str, Any]],
    interpolation: float,
) -> None:
    allocations = result.get("allocations")
    if not isinstance(allocations, list) or len(allocations) != len(action):
        raise ValueError("NeDA interpolation source allocation alignment drift")
    thought_by_turn = {
        (str(record["episode_id"]), int(record["turn_id"])): record
        for record in thought
    }
    action_by_turn = {
        (str(record["episode_id"]), int(record["turn_id"])): record
        for record in action
    }
    if len(thought_by_turn) != len(thought) or len(action_by_turn) != len(action):
        raise ValueError("NeDA interpolation contains duplicate turn records")
    maximum_mass_error = 0.0
    used_thought = set()
    used_action = set()
    for allocation in allocations:
        episode_id = str(allocation.get("episode_id"))
        turn_id = int(allocation.get("turn_id"))
        key = (episode_id, turn_id)
        action_record = action_by_turn.get(key)
        if action_record is None:
            raise ValueError("NeDA interpolation lacks an Action turn record")
        used_action.add(key)
        turn_mass = _finite(allocation.get("turn_mass"), "turn mass")
        action_record["reward"] = turn_mass
        action_record["adv_map"] = [
            turn_mass
        ] * len(action_record["decision_traces"]["action"]["response_ids"])
        action_record["step_credit_by_round"] = {}
        if allocation.get("decision_case") == "ACTION_ONLY":
            if key in thought_by_turn:
                raise ValueError(
                    "Action-only NeDA allocation unexpectedly has Thought data"
                )
            diagnostic = allocation.get("diagnostic")
            if not isinstance(diagnostic, dict) or diagnostic.get(
                "allocation_status"
            ) != "ACTION_ONLY":
                raise ValueError(
                    "Action-only NeDA allocation lacks its audit diagnostic"
                )
            diagnostic["source_interpolation"] = diagnostic.get(
                "interpolation"
            )
            diagnostic["interpolation"] = float(interpolation)
            continue
        thought_record = thought_by_turn.get(key)
        if thought_record is None:
            raise ValueError("NeDA interpolation lacks a Thought turn record")
        used_thought.add(key)
        diagnostic = allocation.get("diagnostic")
        if not isinstance(diagnostic, dict):
            raise ValueError("NeDA interpolation source lacks diagnostics")
        uniform = diagnostic.get("uniform_weights")
        evidence = diagnostic.get("evidence_weights")
        if not isinstance(uniform, dict) or not isinstance(evidence, dict):
            raise ValueError(
                "NeDA interpolation source lacks uniform/evidence weights"
            )
        uniform_weights = {
            int(key): _finite(value, "uniform weight")
            for key, value in uniform.items()
        }
        evidence_weights = {
            int(key): _finite(value, "evidence weight")
            for key, value in evidence.items()
        }
        if set(uniform_weights) != set(evidence_weights):
            raise ValueError("NeDA interpolation simplex support drift")
        weights = {
            round_id: (
                (1.0 - float(interpolation)) * uniform_weights[round_id]
                + float(interpolation) * evidence_weights[round_id]
            )
            for round_id in uniform_weights
        }
        weight_error = abs(sum(weights.values()) - 1.0)
        if weight_error > 1e-10 or any(value < 0.0 for value in weights.values()):
            raise ValueError("NeDA interpolation weights are not a simplex")
        step_credit = {
            round_id: turn_mass * weight
            for round_id, weight in weights.items()
        }
        mass_error = abs(sum(step_credit.values()) - turn_mass)
        maximum_mass_error = max(maximum_mass_error, mass_error)
        trace = thought_record["decision_traces"]["thought"]
        step_map = trace.get("step_map")
        if not isinstance(step_map, list) or not step_map:
            raise ValueError("NeDA interpolation Thought step-map drift")
        thought_record["reward"] = turn_mass
        thought_record["adv_map"] = [
            step_credit.get(int(round_id), 0.0) for round_id in step_map
        ]
        thought_record["step_credit_by_round"] = {
            str(key): value for key, value in step_credit.items()
        }
        allocation["step_credit_by_round"] = {
            str(key): value for key, value in step_credit.items()
        }
        diagnostic["source_interpolation"] = diagnostic.get("interpolation")
        diagnostic["interpolation"] = float(interpolation)
        diagnostic["mass_error"] = mass_error
    if used_thought != set(thought_by_turn) or used_action != set(action_by_turn):
        raise ValueError("NeDA interpolation left unmatched learner records")
    if maximum_mass_error > 1e-9:
        raise ValueError("NeDA interpolation is not mass conserving")
    result["neda_interpolation"] = float(interpolation)
    result["mass_conservation"] = {
        "episode_to_turn": "A_i / H_i",
        "turn_to_denoising": (
            "uniform" if interpolation == 0.0 else "Action evidence only"
        ),
        "maximum_turn_mass_error": maximum_mass_error,
    }


def _randomize_one_position_path(
    thought: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    for record in thought:
        trace = record["decision_traces"]["thought"]
        native = trace.get("native_replay", {})
        position_trace = native.get("position_trace", [])
        for decision in position_trace:
            candidates = [
                int(value) for value in decision.get("candidate_positions", [])
            ]
            selected = [
                int(value) for value in decision.get("selected_positions", [])
            ]
            alternatives = [value for value in candidates if value not in selected]
            if selected and alternatives:
                before = list(selected)
                selected[-1] = alternatives[0]
                if len(set(selected)) != len(selected):
                    continue
                decision["selected_positions"] = selected
                return {
                    "sample_id": str(record["sample_id"]),
                    "round_id": int(decision["round_id"]),
                    "candidate_positions": candidates,
                    "selected_positions_before": before,
                    "selected_positions_after": selected,
                    "stored_behavior_logprob_unchanged": True,
                }
    raise ValueError("no non-forced position decision exists for random-path control")


def materialize_variant(
    variant: str,
    artifact: Mapping[str, Any],
    thought_records: Sequence[Mapping[str, Any]],
    action_records: Sequence[Mapping[str, Any]],
    *,
    activate_online: bool = False,
    online_iteration: int = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    variant = str(variant)
    if variant not in VARIANTS:
        raise ValueError("unknown NeDA ablation variant")
    _validate_source(artifact, thought_records, action_records)
    if activate_online:
        if variant == "random_path":
            raise ValueError(
                "random-path is a diagnostic negative control and cannot be "
                "activated for online training"
            )
        if artifact.get("scientific_result") is not True or artifact.get(
            "engineering_advantage_override", False
        ):
            raise ValueError(
                "online ablation activation requires real scientific credit"
            )
        if online_iteration is None or int(online_iteration) < 0:
            raise ValueError(
                "online ablation activation requires a nonnegative iteration"
            )
    result = copy.deepcopy(dict(artifact))
    thought = [copy.deepcopy(dict(value)) for value in thought_records]
    action = [copy.deepcopy(dict(value)) for value in action_records]
    expected_gate_status = "PASS"
    learner_method = "neda"
    diagnostic = None

    if variant in ("uniform", "evidence_only"):
        target = 0.0 if variant == "uniform" else 1.0
        _reweight_neda(result, thought, action, target)
    elif variant in GENERIC_METHOD:
        learner_method = GENERIC_METHOD[variant]
        for record in thought:
            _rename_record(record, learner_method, "thought")
        for record in action:
            _rename_record(record, learner_method, "action")
        result["method_id"] = learner_method
        if variant == "token_only":
            for record in action:
                record["reward"] = 0.0
                record["adv_map"] = [0.0] * len(record["adv_map"])
                record["step_credit_by_round"] = {}
    elif variant == "no_horizon":
        for record in thought + action:
            horizon = int(record["episode_horizon"])
            if horizon <= 0:
                raise ValueError("no-horizon ablation found invalid horizon")
            _scale_record(record, float(horizon))
        for allocation in result.get("allocations", []):
            horizon = int(allocation["episode_horizon"])
            allocation["turn_mass"] = _finite(
                allocation["turn_mass"], "turn mass"
            ) * float(horizon)
            allocation["step_credit_by_round"] = {
                str(key): _finite(value, "allocation step credit")
                * float(horizon)
                for key, value in allocation.get(
                    "step_credit_by_round", {}
                ).items()
            }
        result["mass_conservation"] = {
            "episode_to_turn": (
                "ablation: copy A_i to every turn; total episode coefficient "
                "mass scales with H_i"
            ),
            "turn_to_denoising": result.get("mass_conservation", {}).get(
                "turn_to_denoising"
            ),
            "maximum_turn_mass_error": 0.0,
            "intentionally_violates_episode_mass_conservation": True,
        }
    elif variant == "random_path":
        diagnostic = _randomize_one_position_path(thought)
        expected_gate_status = "FAIL_EXACT_PATH"

    result["ablation_contract_version"] = CONTRACT_VERSION
    result["ablation_id"] = variant
    result["learner_registered_method"] = learner_method
    result["expected_replay_gate_status"] = expected_gate_status
    result["scientific_result"] = bool(activate_online)
    result["ablation_preparation_only"] = not bool(activate_online)
    result["online_activation"] = (
        {
            "status": "ACTIVE",
            "iteration": int(online_iteration),
            "source_credit_scientific_result": True,
            "engineering_advantage_override": False,
            "independent_on_policy_refresh_required_after_update": True,
        }
        if activate_online else None
    )
    body = copy.deepcopy(result)
    body.pop("artifact_sha256", None)
    result["artifact_sha256"] = sha256_json(body)

    receipt: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "status": "PASS",
        "ablation_id": variant,
        "learner_registered_method": learner_method,
        "expected_replay_gate_status": expected_gate_status,
        "trainable": variant != "random_path",
        "scientific_result": bool(activate_online),
        "online_iteration": (
            int(online_iteration) if activate_online else None
        ),
        "first_iteration_may_share_authenticated_behavior_rollout": True,
        "independent_on_policy_refresh_required_after_update": (
            variant != "random_path"
        ),
        "thought_records": len(thought),
        "action_records": len(action),
        "nonzero_thought_coefficients": sum(
            abs(float(value)) > 1e-12
            for record in thought for value in record["adv_map"]
        ),
        "nonzero_action_coefficients": sum(
            abs(float(value)) > 1e-12
            for record in action for value in record["adv_map"]
        ),
        "random_path_diagnostic": diagnostic,
        "interpretation": (
            "activated on-policy ablation input for one registered online "
            "iteration"
            if activate_online else
            "artifact preparation only; no optimizer, checkpoint, evaluation, "
            "or method result"
        ),
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    return result, thought, action, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--credit", required=True)
    parser.add_argument("--thought-data", required=True)
    parser.add_argument("--action-data", required=True)
    parser.add_argument("--out-credit", required=True)
    parser.add_argument("--out-thought-data", required=True)
    parser.add_argument("--out-action-data", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--activate-online", action="store_true")
    parser.add_argument("--online-iteration", type=int)
    args = parser.parse_args()

    artifact = _load(args.credit)
    thought = _load(args.thought_data)
    action = _load(args.action_data)
    result, out_thought, out_action, receipt = materialize_variant(
        args.variant,
        artifact,
        thought,
        action,
        activate_online=args.activate_online,
        online_iteration=args.online_iteration,
    )
    _atomic_json(result, args.out_credit)
    _atomic_json(out_thought, args.out_thought_data)
    _atomic_json(out_action, args.out_action_data)
    gate = {
        "contract_version": GATE_VERSION,
        "status": (
            "PASS"
            if receipt["expected_replay_gate_status"] == "PASS"
            else "NEGATIVE_CONTROL_READY"
        ),
        "ablation_id": args.variant,
        "learner_registered_method": receipt["learner_registered_method"],
        "policy_update_eligible": bool(receipt["trainable"]),
        "scientific_result": bool(receipt["scientific_result"]),
        "online_iteration": receipt["online_iteration"],
        "expected_replay_gate_status": receipt[
            "expected_replay_gate_status"
        ],
        "source_credit": {
            "path": os.path.realpath(args.credit),
            "sha256": sha256_file(args.credit),
        },
        "credit": {
            "path": os.path.realpath(args.out_credit),
            "sha256": sha256_file(args.out_credit),
        },
        "thought_data": {
            "path": os.path.realpath(args.out_thought_data),
            "sha256": sha256_file(args.out_thought_data),
        },
        "action_data": {
            "path": os.path.realpath(args.out_action_data),
            "sha256": sha256_file(args.out_action_data),
        },
        "receipt": receipt,
    }
    gate["gate_sha256"] = sha256_json(gate)
    _atomic_json(gate, args.gate)
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
