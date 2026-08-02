#!/usr/bin/env python3
"""Fail-closed structural and reliability audit for V4-O02/O03 artifacts."""

import argparse
import json
import math
import os
from typing import Any, Dict, List, Mapping, Sequence

from neda_counterfactual import (
    MEMORY_CONTRACT,
    spearman_correlation,
    summarize_paired_effects,
    validate_crn_schedule,
)
from neda_repro import sha256_file, sha256_json
from neda_step_token_counterfactual import (
    REPLACEMENT_CONTRACT,
    STEP_TOKEN_CONTRACT_VERSION,
)


AUDIT_CONTRACT_VERSION = "neda-o02-o03-gate-v2"


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("O02 artifact must be a JSON object")
    return value


def _same_summary(stored: Mapping[str, Any], rebuilt: Mapping[str, Any]) -> bool:
    keys = (
        "n_pairs",
        "original_mean_return",
        "alternative_mean_return",
        "effect_samples",
        "mean_effect",
    )
    return all(stored.get(key) == rebuilt.get(key) for key in keys)


def audit_step_token_artifacts(
    artifacts: Sequence[Mapping[str, Any]], expected_games: int = 0
) -> Dict[str, Any]:
    if not artifacts:
        raise ValueError("at least one O02 artifact is required")
    games = set()
    anchors_seen = set()
    retest_anchors = set()
    base_hashes = set()
    model_hashes = set()
    replacement_hashes = set()
    schedule_hashes = set()
    reference_count = 0
    pair_count = 0
    step_reference_count = 0
    token_reference_count = 0
    action_token_reference_count = 0
    action_changed = 0
    action_token_pairs = 0
    action_token_changed = 0
    max_logprob_error = 0.0
    repeat_zero: Dict[str, float] = {}
    repeat_one: Dict[str, float] = {}

    for artifact in artifacts:
        if artifact.get("contract_version") != STEP_TOKEN_CONTRACT_VERSION:
            raise ValueError("unsupported O02/O03 artifact contract")
        if artifact.get("artifact_kind") != "step-token-branch-results":
            raise ValueError("unexpected O02/O03 artifact kind")
        if not artifact.get("complete"):
            raise ValueError("O02/O03 artifact is incomplete")
        if artifact.get("memory_contract") != MEMORY_CONTRACT:
            raise ValueError("O02/O03 memory contract drift")
        protocol = artifact.get("protocol", {})
        if protocol.get("replacement_contract") != REPLACEMENT_CONTRACT:
            raise ValueError("O02/O03 replacement contract drift")
        k = int(protocol.get("k", 0))
        n_steps = int(protocol.get("n_steps", 0))
        tokens_per_step = int(protocol.get("tokens_per_step", 0))
        max_action_tokens = int(protocol.get("max_action_tokens", 0))
        min_action_tokens = int(protocol.get("min_action_tokens", 0))
        if (
            k < 1
            or n_steps != 2
            or tokens_per_step != 2
            or max_action_tokens != 2
            or min_action_tokens != 1
        ):
            raise ValueError("O02/O03 registered 2x2xK design drift")
        game_id = str(artifact.get("game_id", ""))
        if not game_id or game_id in games:
            raise ValueError("duplicate or empty O02 game shard")
        games.add(game_id)
        base_hashes.add(str(artifact.get("base_artifact_sha256", "")))
        model_hashes.add(str(artifact.get("model_identity_sha256", "")))
        local_retest = set(str(value) for value in artifact.get("retest_anchor_ids", []))
        anchors = list(artifact.get("anchors", []))
        if len(anchors) != 6:
            raise ValueError("each O02 game shard must contain six anchors")
        for anchor in anchors:
            anchor_id = str(anchor.get("anchor_id", ""))
            if not anchor_id or anchor_id in anchors_seen:
                raise ValueError("duplicate or empty O02 anchor")
            anchors_seen.add(anchor_id)
            expected_repeat_ids = {"0", "1"} if anchor_id in local_retest else {"0"}
            repeats = anchor.get("repeats", {})
            if set(repeats) != expected_repeat_ids:
                raise ValueError("O03 retest membership/count drift")
            if anchor_id in local_retest:
                retest_anchors.add(anchor_id)
            selection = anchor.get("selection", {})
            thought_selection = selection.get("thought", {})
            action_selection = selection.get("action", {})
            if len(thought_selection.get("steps", [])) != n_steps:
                raise ValueError("anchor selection step count drift")
            action_coordinates = list(action_selection.get("coordinates", []))
            if not (
                min_action_tokens <= len(action_coordinates) <= max_action_tokens
            ):
                raise ValueError("anchor Action-token selection count drift")
            expected_coordinates = (
                n_steps * (1 + tokens_per_step) + len(action_coordinates)
            )
            for repeat_id, repeat in repeats.items():
                originals = repeat.get("originals", {})
                if set(originals) != {str(index) for index in range(k)}:
                    raise ValueError("shared original sample count drift")
                references = list(repeat.get("references", []))
                if len(references) != expected_coordinates:
                    raise ValueError("coordinate reference count drift")
                coordinate_ids = [str(row.get("coordinate_id", "")) for row in references]
                if len(set(coordinate_ids)) != len(coordinate_ids):
                    raise ValueError("duplicate coordinate reference")
                if [row.get("level") for row in references].count("step") != n_steps:
                    raise ValueError("step reference count drift")
                if [row.get("level") for row in references].count("token") != n_steps * tokens_per_step:
                    raise ValueError("token reference count drift")
                if [row.get("level") for row in references].count("action_token") != len(action_coordinates):
                    raise ValueError("Action-token reference count drift")
                for reference in references:
                    level = str(reference.get("level", ""))
                    if level not in ("step", "token", "action_token"):
                        raise ValueError("invalid coordinate level")
                    pairs = list(reference.get("pairs", []))
                    if len(pairs) != k:
                        raise ValueError("coordinate K completion count drift")
                    rebuilt_pairs = []
                    for sample_id, pair in enumerate(pairs):
                        if int(pair.get("sample_id", -1)) != sample_id:
                            raise ValueError("pair sample IDs are not contiguous")
                        schedule = pair.get("crn_schedule", {})
                        validate_crn_schedule(schedule)
                        if schedule.get("anchor_id") != anchor_id or int(schedule.get("repeat_id", -1)) != int(repeat_id):
                            raise ValueError("CRN schedule anchor/repeat drift")
                        schedule_hashes.add(str(schedule["schedule_sha256"]))
                        original = originals.get(str(sample_id), {})
                        alternative = pair.get("alternative", {})
                        for role, branch in (("original", original), ("alternative", alternative)):
                            if branch.get("crn_schedule_sha256") != schedule["schedule_sha256"]:
                                raise ValueError("{} branch CRN drift".format(role))
                            prefix = branch.get("prefix_replay", {})
                            if prefix.get("status") != "PASS":
                                raise ValueError("{} branch prefix replay failed".format(role))
                        if original.get("original_anchor_transition_reproduced") is not True:
                            raise ValueError("original anchor transition was not reproduced")
                        replacement = pair.get("replacement", {})
                        if replacement.get("contract") != REPLACEMENT_CONTRACT:
                            raise ValueError("replacement contract drift")
                        if replacement.get("coordinate_id") != reference.get("coordinate_id"):
                            raise ValueError("replacement coordinate drift")
                        originals_ids = list(replacement.get("original_token_ids", []))
                        replacement_ids = list(replacement.get("replacement_token_ids", []))
                        if not originals_ids or len(originals_ids) != len(replacement_ids):
                            raise ValueError("replacement token shape drift")
                        if any(left == right for left, right in zip(originals_ids, replacement_ids)):
                            raise ValueError("excluded realized token was resampled")
                        replacement_sha = str(replacement.get("replacement_sha256", ""))
                        without_sha = dict(replacement)
                        without_sha.pop("replacement_sha256", None)
                        if replacement_sha != sha256_json(without_sha):
                            raise ValueError("replacement SHA drift")
                        if replacement_sha in replacement_hashes:
                            raise ValueError("duplicate coordinate/sample replacement")
                        replacement_hashes.add(replacement_sha)
                        error = float(replacement.get("max_abs_behavior_logprob_error", math.inf))
                        if not math.isfinite(error) or error > float(protocol["logprob_tolerance"]):
                            raise ValueError("recorded-state logprob audit failed")
                        max_logprob_error = max(max_logprob_error, error)
                        regenerated = pair.get("regenerated_action", {})
                        if not str(regenerated.get("action", "")):
                            raise ValueError("regenerated Action is empty")
                        if pair.get("action_changed"):
                            action_changed += 1
                        if level == "action_token":
                            action_token_pairs += 1
                            if pair.get("action_changed"):
                                action_token_changed += 1
                            else:
                                raise ValueError(
                                    "Action-token intervention did not change Action"
                                )
                        stored_effect = float(pair.get("paired_effect"))
                        effect = float(original["return"]) - float(alternative["return"])
                        if abs(stored_effect - effect) > 1e-12:
                            raise ValueError("paired effect drift")
                        rebuilt_pairs.append(
                            {
                                "crn_schedule": schedule,
                                "original": original,
                                "alternative": alternative,
                                "paired_effect": effect,
                            }
                        )
                        pair_count += 1
                    rebuilt = summarize_paired_effects(rebuilt_pairs)
                    if not _same_summary(reference.get("reference", {}), rebuilt):
                        raise ValueError("stored coordinate summary drift")
                    reliability_key = "{}:{}".format(anchor_id, reference["coordinate_id"])
                    if repeat_id == "0":
                        repeat_zero[reliability_key] = float(rebuilt["mean_effect"])
                    else:
                        repeat_one[reliability_key] = float(rebuilt["mean_effect"])
                    reference_count += 1
                    if level == "step":
                        step_reference_count += 1
                    elif level == "token":
                        token_reference_count += 1
                    else:
                        action_token_reference_count += 1

    if len(base_hashes) != 1 or "" in base_hashes:
        raise ValueError("O02 shards do not share one frozen O01 base")
    if len(model_hashes) != 1 or "" in model_hashes:
        raise ValueError("O02 shards do not share one behavior checkpoint")
    if expected_games and len(games) != int(expected_games):
        raise ValueError(
            "expected {} O02 games, found {}".format(expected_games, len(games))
        )
    if int(expected_games) == 16:
        if len(anchors_seen) != 96 or len(retest_anchors) != 24:
            raise ValueError("full O02/O03 requires 96 anchors and 24 retest anchors")
    paired_keys = sorted(set(repeat_zero) & set(repeat_one))
    test_retest = {
        "n_coordinates": len(paired_keys),
        "spearman": spearman_correlation(
            [repeat_zero[key] for key in paired_keys],
            [repeat_one[key] for key in paired_keys],
        ),
    }
    result = {
        "contract_version": AUDIT_CONTRACT_VERSION,
        "status": "PASS",
        "phase": "o02_o03",
        "n_games": len(games),
        "n_anchors": len(anchors_seen),
        "n_retest_anchors": len(retest_anchors),
        "n_references": reference_count,
        "n_step_references": step_reference_count,
        "n_token_references": token_reference_count,
        "n_thought_token_references": token_reference_count,
        "n_action_token_references": action_token_reference_count,
        "n_pairs": pair_count,
        "n_unique_schedules": len(schedule_hashes),
        "n_unique_replacements": len(replacement_hashes),
        "action_change_rate": action_changed / float(pair_count),
        "action_token_action_change_rate": action_token_changed
        / float(action_token_pairs),
        "max_abs_behavior_logprob_error": max_logprob_error,
        "base_artifact_sha256": next(iter(base_hashes)),
        "model_identity_sha256": next(iter(model_hashes)),
        "test_retest": test_retest,
    }
    result["audit_sha256"] = sha256_json(result)
    return result


def atomic_json(value: Mapping[str, Any], path: str) -> None:
    path = os.path.realpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", nargs="+", required=True)
    parser.add_argument("--expected-games", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    documents = [_load(path) for path in args.artifacts]
    result = audit_step_token_artifacts(documents, args.expected_games)
    result["artifacts"] = [
        {"path": os.path.realpath(path), "sha256": sha256_file(path)}
        for path in args.artifacts
    ]
    # The audit SHA covers scientific fields; file receipts are appended after
    # it so relocating an otherwise identical shard does not change statistics.
    atomic_json(result, args.out)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
