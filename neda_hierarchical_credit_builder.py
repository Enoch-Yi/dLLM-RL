#!/usr/bin/env python3
"""Build the audited V4-E05 OOF hierarchical-credit artifact.

This is an engineering and attribution gate, not a policy-performance result.
It consumes the frozen O01/O02 counterfactual references, fits game-group OOF
ridge heads for turn, Thought-step, Thought-token, and Action-token effects,
and materializes mass-conserving R/H/D/T credit on every recorded base turn.
"""

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from neda_counterfactual_audit import (
    audit_counterfactual_artifact,
    merge_counterfactual_artifacts,
)
from neda_hierarchical_credit import (
    HIERARCHY_CONTRACT_VERSION,
    allocate_hierarchy,
    audit_hierarchy,
    contract_probes,
    grouped_oof_ridge,
)
from neda_repro import sha256_file, sha256_json
from neda_step_token_audit import (
    AUDIT_CONTRACT_VERSION as STEP_TOKEN_AUDIT_CONTRACT_VERSION,
    audit_step_token_artifacts,
)


BUILDER_CONTRACT_VERSION = "neda-e05-hierarchical-credit-builder-v1"
GATE_CONTRACT_VERSION = "neda-e05-hierarchical-credit-gate-v1"

TURN_FEATURES = [
    "turn_fraction",
    "remaining_fraction",
    "log1p_horizon",
    "episode_return",
    "cumulative_reward_before",
    "turn_reward",
    "log1p_admissible_count",
    "log1p_thought_tokens",
    "log1p_action_tokens",
    "thought_mean_nll",
    "action_mean_nll",
    "thought_commit_fraction",
]

STEP_FEATURES = TURN_FEATURES + [
    "step_fraction",
    "commit_token_fraction",
    "commit_mean_nll",
    "commit_max_nll",
    "commit_mean_confidence",
]

THOUGHT_TOKEN_FEATURES = TURN_FEATURES + [
    "step_fraction",
    "token_fraction",
    "token_nll",
    "token_confidence",
    "commit_token_fraction",
]

ACTION_TOKEN_FEATURES = TURN_FEATURES + [
    "token_fraction",
    "token_nll",
    "token_confidence",
    "prefix_fraction",
    "log1p_action_width",
]


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON object required: {}".format(path))
    return value


def _atomic_json(value: Mapping[str, Any], path: str) -> None:
    path = os.path.realpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_mean(values: Sequence[float]) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def _finite_vector(values: Sequence[float], label: str) -> List[float]:
    result = [float(value) for value in values]
    if any(not math.isfinite(value) for value in result):
        raise ValueError("{} features contain non-finite values".format(label))
    return result


def _trace(turn: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    trace = turn.get("decision_traces", {}).get(kind)
    if not isinstance(trace, Mapping) or not trace.get("response_ids"):
        raise ValueError("base turn has no non-empty {} trace".format(kind))
    n = len(trace["response_ids"])
    if any(len(trace.get(key, [])) != n for key in ("step_map", "behavior_logprobs")):
        raise ValueError("{} trace alignment drift".format(kind))
    return trace


def turn_feature_vector(
    episode: Mapping[str, Any], turn: Mapping[str, Any]
) -> List[float]:
    horizon = int(episode["horizon"])
    turn_id = int(turn["turn_id"])
    thought = _trace(turn, "thought")
    action = _trace(turn, "action")
    thought_nll = [-float(value) for value in thought["behavior_logprobs"]]
    action_nll = [-float(value) for value in action["behavior_logprobs"]]
    commits = len(set(int(value) for value in thought["step_map"]))
    denominator = max(1, horizon - 1)
    state = turn.get("state_before", {})
    # ALFWorld and WebShop expose the same finite Action interface under
    # different field names.  Keep one frozen feature schema while reading
    # the environment-native state record; silently treating every WebShop
    # state as having zero admissible Actions would create a benchmark-only
    # covariate bug in the cross-fitted credit heads.
    admissible = state.get("admissible_commands")
    if admissible is None:
        admissible = state.get("admissible_actions", [])
    cumulative_reward = state.get("cumulative_reward")
    if cumulative_reward is None:
        cumulative_reward = state.get("progress", 0.0)
    values = [
        turn_id / float(denominator),
        (horizon - 1 - turn_id) / float(denominator),
        math.log1p(horizon),
        float(episode["return"]),
        float(cumulative_reward),
        float(turn.get("turn_reward", 0.0)),
        math.log1p(len(admissible)),
        math.log1p(len(thought["response_ids"])),
        math.log1p(len(action["response_ids"])),
        _safe_mean(thought_nll),
        _safe_mean(action_nll),
        commits / float(max(1, len(thought["response_ids"]))),
    ]
    return _finite_vector(values, "turn")


def _positions_for_step(trace: Mapping[str, Any], step_id: int) -> List[int]:
    positions = [
        index
        for index, value in enumerate(trace["step_map"])
        if int(value) == int(step_id)
    ]
    if not positions:
        raise ValueError("selected Thought step is absent from the trace")
    return positions


def step_feature_vector(
    episode: Mapping[str, Any],
    turn: Mapping[str, Any],
    step_id: int,
    positions: Sequence[int] = (),
) -> List[float]:
    trace = _trace(turn, "thought")
    positions = list(positions) or _positions_for_step(trace, step_id)
    if any(int(trace["step_map"][position]) != int(step_id) for position in positions):
        raise ValueError("Thought-step feature positions cross commit steps")
    max_step = max(int(value) for value in trace["step_map"])
    nll = [-float(trace["behavior_logprobs"][position]) for position in positions]
    confidence = trace.get("commit_confidence", [])
    conf = (
        [float(confidence[position]) for position in positions]
        if len(confidence) == len(trace["response_ids"])
        else [math.exp(-value) for value in nll]
    )
    values = turn_feature_vector(episode, turn) + [
        int(step_id) / float(max(1, max_step)),
        len(positions) / float(len(trace["response_ids"])),
        _safe_mean(nll),
        max(nll),
        _safe_mean(conf),
    ]
    return _finite_vector(values, "Thought-step")


def thought_token_feature_vector(
    episode: Mapping[str, Any], turn: Mapping[str, Any], position: int
) -> List[float]:
    trace = _trace(turn, "thought")
    position = int(position)
    if not (0 <= position < len(trace["response_ids"])):
        raise ValueError("Thought-token position is outside the trace")
    step_id = int(trace["step_map"][position])
    positions = _positions_for_step(trace, step_id)
    max_step = max(int(value) for value in trace["step_map"])
    confidence = trace.get("commit_confidence", [])
    token_nll = -float(trace["behavior_logprobs"][position])
    token_confidence = (
        float(confidence[position])
        if len(confidence) == len(trace["response_ids"])
        else math.exp(-token_nll)
    )
    values = turn_feature_vector(episode, turn) + [
        step_id / float(max(1, max_step)),
        position / float(max(1, len(trace["response_ids"]) - 1)),
        token_nll,
        token_confidence,
        len(positions) / float(len(trace["response_ids"])),
    ]
    return _finite_vector(values, "Thought-token")


def action_token_feature_vector(
    episode: Mapping[str, Any], turn: Mapping[str, Any], position: int
) -> List[float]:
    trace = _trace(turn, "action")
    position = int(position)
    if not (0 <= position < len(trace["response_ids"])):
        raise ValueError("Action-token position is outside the trace")
    confidence = trace.get("commit_confidence", [])
    token_nll = -float(trace["behavior_logprobs"][position])
    token_confidence = (
        float(confidence[position])
        if len(confidence) == len(trace["response_ids"])
        else math.exp(-token_nll)
    )
    width = int(trace.get("replay_width", len(trace["response_ids"])))
    values = turn_feature_vector(episode, turn) + [
        position / float(max(1, len(trace["response_ids"]) - 1)),
        token_nll,
        token_confidence,
        position / float(max(1, width)),
        math.log1p(width),
    ]
    return _finite_vector(values, "Action-token")


def fit_oof_head(
    name: str,
    feature_names: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    folds: int = 4,
    ridge: float = 10.0,
    seed: int = 92001,
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("{} head has no counterfactual targets".format(name))
    identifiers = [str(row["coordinate_id"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("{} head has duplicate coordinates".format(name))
    if any(len(row["features"]) != len(feature_names) for row in rows):
        raise ValueError("{} feature schema drift".format(name))
    result = grouped_oof_ridge(
        [row["features"] for row in rows],
        [float(row["target"]) for row in rows],
        [str(row["game_id"]) for row in rows],
        folds=folds,
        ridge=ridge,
        seed=seed,
    )
    result.update(
        {
            "head": name,
            "feature_names": list(feature_names),
            "n_features": len(feature_names),
            "n_targets": len(rows),
            "r2_role": "diagnostic-only-no-pass-threshold",
            "rows": [
                {
                    "coordinate_id": identifiers[index],
                    "game_id": str(row["game_id"]),
                    "target": float(row["target"]),
                    "prediction": float(result["predictions"][index]),
                    "fold": int(result["fold_assignments"][index]),
                    "reference_se": row.get("reference_se"),
                }
                for index, row in enumerate(rows)
            ],
        }
    )
    result["head_sha256"] = sha256_json(result)
    return result


def predict_oof(head: Mapping[str, Any], features: Sequence[float], game_id: str) -> float:
    features = _finite_vector(features, "OOF prediction")
    if len(features) != int(head["n_features"]):
        raise ValueError("OOF prediction feature width drift")
    fold_map = head["fold_map"]
    if str(game_id) not in fold_map:
        raise ValueError("OOF prediction game is outside frozen fold map")
    fold = int(fold_map[str(game_id)])
    model = next(row for row in head["fold_rows"] if int(row["fold"]) == fold)
    means = [float(value) for value in model["feature_mean"]]
    scales = [float(value) for value in model["feature_scale"]]
    weights = [float(value) for value in model["weights"]]
    if len(weights) != len(features) + 1:
        raise ValueError("OOF model coefficient width drift")
    value = weights[-1] + sum(
        ((feature - mean) / scale) * weight
        for feature, mean, scale, weight in zip(features, means, scales, weights[:-1])
    )
    if not math.isfinite(value):
        raise ValueError("OOF prediction is non-finite")
    return float(value)


def _episode_index(base: Mapping[str, Any]):
    episodes = {str(row["episode_id"]): row for row in base["base_episodes"]}
    turns = {}
    for episode_id, episode in episodes.items():
        if int(episode["horizon"]) != len(episode["turns"]):
            raise ValueError("base episode horizon drift")
        for turn in episode["turns"]:
            key = (episode_id, int(turn["turn_id"]))
            if key in turns:
                raise ValueError("duplicate base turn coordinate")
            turns[key] = (episode, turn)
    return episodes, turns


def build_target_rows(
    base: Mapping[str, Any],
    o01: Mapping[str, Any],
    o02_shards: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    _, turns = _episode_index(base)
    result: Dict[str, List[Dict[str, Any]]] = {
        "turn": [],
        "thought_step": [],
        "thought_token": [],
        "action_token": [],
    }
    for anchor in o01["anchors"]:
        episode, turn = turns[(str(anchor["episode_id"]), int(anchor["turn_id"]))]
        reference = anchor["repeats"]["0"]["reference"]
        result["turn"].append(
            {
                "coordinate_id": str(anchor["anchor_id"]),
                "game_id": str(anchor["game_id"]),
                "features": turn_feature_vector(episode, turn),
                "target": float(reference["mean_effect"]),
                "reference_se": reference.get("standard_error"),
            }
        )
    for shard in o02_shards:
        for anchor in shard["anchors"]:
            episode, turn = turns[(str(anchor["episode_id"]), int(anchor["turn_id"]))]
            for reference in anchor["repeats"]["0"]["references"]:
                level = str(reference["level"])
                target = float(reference["reference"]["mean_effect"])
                common = {
                    "coordinate_id": str(reference["coordinate_id"]),
                    "game_id": str(anchor["game_id"]),
                    "target": target,
                    "reference_se": reference["reference"].get("standard_error"),
                }
                if level == "step":
                    common["features"] = step_feature_vector(
                        episode,
                        turn,
                        int(reference["step_id"]),
                        reference["positions"],
                    )
                    result["thought_step"].append(common)
                elif level == "token":
                    common["features"] = thought_token_feature_vector(
                        episode, turn, int(reference["positions"][0])
                    )
                    result["thought_token"].append(common)
                elif level == "action_token":
                    common["features"] = action_token_feature_vector(
                        episode, turn, int(reference["positions"][0])
                    )
                    result["action_token"].append(common)
                else:
                    raise ValueError("unsupported O02 coordinate level")
    return result


def _group_advantages(base: Mapping[str, Any]) -> Dict[str, float]:
    by_game = defaultdict(list)
    for episode in base["base_episodes"]:
        by_game[str(episode["game_id"])].append(episode)
    values = {}
    for episodes in by_game.values():
        returns = [float(row["return"]) for row in episodes]
        mean = statistics.mean(returns)
        scale = statistics.pstdev(returns) if len(returns) > 1 else 0.0
        for episode in episodes:
            values[str(episode["episode_id"])] = (
                (float(episode["return"]) - mean) / (scale + 1e-6)
                if scale > 0.0
                else 0.0
            )
    return values


def materialize_hierarchies(
    base: Mapping[str, Any], heads: Mapping[str, Mapping[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    advantages = _group_advantages(base)
    episodes_out = []
    maximum = {"turn": 0.0, "step": 0.0, "token": 0.0}
    for episode in base["base_episodes"]:
        game_id = str(episode["game_id"])
        turn_scores = []
        step_scores = []
        token_scores = []
        metadata = []
        for turn in episode["turns"]:
            turn_scores.append(
                predict_oof(heads["turn"], turn_feature_vector(episode, turn), game_id)
            )
            local_step_scores = []
            local_token_scores = []
            local_metadata = []
            thought = _trace(turn, "thought")
            for step_id in sorted(set(int(value) for value in thought["step_map"])):
                positions = _positions_for_step(thought, step_id)
                local_step_scores.append(
                    predict_oof(
                        heads["thought_step"],
                        step_feature_vector(episode, turn, step_id, positions),
                        game_id,
                    )
                )
                local_token_scores.append(
                    [
                        predict_oof(
                            heads["thought_token"],
                            thought_token_feature_vector(episode, turn, position),
                            game_id,
                        )
                        for position in positions
                    ]
                )
                local_metadata.append(
                    {
                        "kind": "thought_denoising_step",
                        "recorded_step_id": step_id,
                        "token_positions": positions,
                    }
                )
            action = _trace(turn, "action")
            action_values = [
                predict_oof(
                    heads["action_token"],
                    action_token_feature_vector(episode, turn, position),
                    game_id,
                )
                for position in range(len(action["response_ids"]))
            ]
            local_step_scores.append(_safe_mean(action_values))
            local_token_scores.append(action_values)
            local_metadata.append(
                {
                    "kind": "environment_action_joint",
                    "recorded_step_id": None,
                    "token_positions": list(range(len(action["response_ids"]))),
                    "parent_score": "mean-action-token-oof-prediction",
                }
            )
            step_scores.append(local_step_scores)
            token_scores.append(local_token_scores)
            metadata.append(local_metadata)
        hierarchy = allocate_hierarchy(
            advantages[str(episode["episode_id"])],
            1.0,
            turn_scores,
            step_scores,
            token_scores,
            eta_h=0.5,
            eta_d=0.5,
            eta_t=0.5,
            outer_prior="uniform",
            gamma=0.95,
        )
        for turn_row, turn, step_meta in zip(
            hierarchy["turns"], episode["turns"], metadata
        ):
            turn_row["turn_id"] = int(turn["turn_id"])
            for step_row, meta in zip(turn_row["steps"], step_meta):
                step_row.update(meta)
                for token_row, position in zip(step_row["tokens"], meta["token_positions"]):
                    token_row["recorded_token_position"] = int(position)
        audit = audit_hierarchy(hierarchy)
        maximum["turn"] = max(maximum["turn"], audit["turn_mass_error"])
        maximum["step"] = max(maximum["step"], audit["max_step_parent_error"])
        maximum["token"] = max(maximum["token"], audit["max_token_parent_error"])
        episodes_out.append(
            {
                "game_id": game_id,
                "episode_id": str(episode["episode_id"]),
                "rollout_id": int(episode["rollout_id"]),
                "return": float(episode["return"]),
                "group_advantage": advantages[str(episode["episode_id"])],
                "hierarchy": hierarchy,
            }
        )
    return episodes_out, maximum


def build(args: argparse.Namespace) -> Dict[str, Any]:
    o02_paths = sorted(
        os.path.join(args.o02_root, "game_{:02d}.json".format(index))
        for index in range(int(args.expected_games))
    )
    missing = [path for path in o02_paths if not os.path.isfile(path)]
    if missing:
        raise ValueError("missing O02 shards: {}".format(missing[:3]))
    o02_shards = [_load(path) for path in o02_paths]
    o02_audit = audit_step_token_artifacts(
        o02_shards, expected_games=int(args.expected_games)
    )
    if o02_audit.get("status") != "PASS":
        raise ValueError("O02/O03 audit is not PASS")
    recorded_gate = _load(args.o02_gate)
    if (
        recorded_gate.get("contract_version") != STEP_TOKEN_AUDIT_CONTRACT_VERSION
        or recorded_gate.get("status") != "PASS"
    ):
        raise ValueError("recorded O02/O03 gate is missing or incompatible")
    if recorded_gate.get("audit_sha256") != o02_audit.get("audit_sha256"):
        raise ValueError("recorded O02/O03 gate differs from rebuilt audit")

    base_paths = {os.path.realpath(str(row["base_artifact"])) for row in o02_shards}
    if len(base_paths) != 1:
        raise ValueError("O02 shards do not identify one O01 base")
    base_path = next(iter(base_paths))
    base = _load(base_path)
    if sha256_file(base_path) != o02_audit["base_artifact_sha256"]:
        raise ValueError("O01 base SHA differs from O02 registration")
    o01_root = os.path.dirname(base_path)
    o01_audit_path = os.path.join(o01_root, "o01_audit.json")
    recorded_o01_audit = _load(o01_audit_path)
    if recorded_o01_audit.get("status") != "PASS":
        raise ValueError("O01 reliability gate is not PASS")
    branch_paths = [
        os.path.join(o01_root, "branch_{:02d}.json".format(index))
        for index in range(int(args.expected_games))
    ]
    branches = [_load(path) for path in branch_paths]
    merged = merge_counterfactual_artifacts(branches)
    rebuilt_o01_audit = audit_counterfactual_artifact(merged, phase="o01")
    if rebuilt_o01_audit.get("status") != "PASS":
        raise ValueError("rebuilt O01 reliability gate is not PASS")
    for field in (
        "n_anchors",
        "n_retest_selected",
        "test_retest",
        "useful_reference_ci_fraction",
    ):
        if recorded_o01_audit.get(field) != rebuilt_o01_audit.get(field):
            raise ValueError("recorded O01 gate drift for {}".format(field))

    targets = build_target_rows(base, merged, o02_shards)
    heads = {
        "turn": fit_oof_head("turn", TURN_FEATURES, targets["turn"]),
        "thought_step": fit_oof_head(
            "thought_step", STEP_FEATURES, targets["thought_step"]
        ),
        "thought_token": fit_oof_head(
            "thought_token", THOUGHT_TOKEN_FEATURES, targets["thought_token"]
        ),
        "action_token": fit_oof_head(
            "action_token", ACTION_TOKEN_FEATURES, targets["action_token"]
        ),
    }
    if any(
        row.get("group_overlap")
        for head in heads.values()
        for row in head["fold_rows"]
    ):
        raise ValueError("OOF head contains game-group leakage")
    hierarchies, maximum_mass_error = materialize_hierarchies(base, heads)
    probes = contract_probes()
    if probes.get("status") != "PASS":
        raise ValueError("hierarchical contract probes failed")
    artifact = {
        "contract_version": BUILDER_CONTRACT_VERSION,
        "hierarchy_contract_version": HIERARCHY_CONTRACT_VERSION,
        "status": "PASS",
        "phase": "V4-E05",
        "scientific_role": "engineering-and-attribution-gate-not-policy-result",
        "base_artifact": base_path,
        "base_artifact_sha256": sha256_file(base_path),
        "o01_root": o01_root,
        "o01_audit": o01_audit_path,
        "o01_audit_sha256": sha256_file(o01_audit_path),
        "o02_gate": os.path.realpath(args.o02_gate),
        "o02_gate_sha256": sha256_file(args.o02_gate),
        "source_shards": [
            {"path": os.path.realpath(path), "sha256": sha256_file(path)}
            for path in o02_paths
        ],
        "design": {
            "fold_unit": "game_id",
            "folds": 4,
            "ridge": 10.0,
            "seed": 92001,
            "mass_constant": 1.0,
            "eta": {"H": 0.5, "D": 0.5, "T": 0.5},
            "action_parent": "joint Action score is mean of Action-token OOF predictions",
            "action_token_role": "primary token audit",
            "thought_token_role": "separate mediated-through-Action token audit",
        },
        "target_counts": {key: len(value) for key, value in targets.items()},
        "heads": heads,
        "episodes": hierarchies,
        "n_episodes": len(hierarchies),
        "maximum_mass_error": maximum_mass_error,
        "contract_probes": probes,
        "o01_rebuilt_gate": rebuilt_o01_audit,
        "o02_rebuilt_gate": o02_audit,
        "claim_boundary": (
            "OOF R2 is diagnostic, not a causal-fidelity result or pass threshold. "
            "Counterfactual target fidelity and downstream online gain remain separate gates."
        ),
    }
    artifact["artifact_sha256"] = sha256_json(artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--o02-root", required=True)
    parser.add_argument("--o02-gate", required=True)
    parser.add_argument("--expected-games", type=int, default=16)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gate", required=True)
    args = parser.parse_args()
    artifact = build(args)
    _atomic_json(artifact, args.out)
    gate = {
        "contract_version": GATE_CONTRACT_VERSION,
        "status": "PASS",
        "phase": "V4-E05",
        "artifact": os.path.realpath(args.out),
        "artifact_sha256": sha256_file(args.out),
        "scientific_artifact_sha256": artifact["artifact_sha256"],
        "n_episodes": artifact["n_episodes"],
        "target_counts": artifact["target_counts"],
        "maximum_mass_error": artifact["maximum_mass_error"],
        "oof_r2_diagnostic": {
            key: value["oof_r2"] for key, value in artifact["heads"].items()
        },
        "game_group_leakage": False,
        "eta_zero_degeneracy": artifact["contract_probes"][
            "eta_zero_uniform_errors"
        ],
    }
    gate["gate_sha256"] = sha256_json(gate)
    _atomic_json(gate, args.gate)
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
