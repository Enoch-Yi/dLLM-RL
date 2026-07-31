#!/usr/bin/env python3
"""Submission-scale ALFWorld counterfactual credit benchmark (V4-2)."""

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Mapping, Sequence

from alfworld_counterfactual import (
    Runtime,
    atomic_json_dump,
    close_env,
    collect_base_episode,
    environment_manager,
    make_repeated_env,
    replace_action_in_generation,
    run_one_branch,
    validate_registered_seeds,
)
from alfworld_step_token_counterfactual import (
    action_from_modified_thought,
    resample_recorded_action_coordinate,
    resample_recorded_thought_coordinate,
    score_action_token_distribution,
)
from neda_counterfactual import MEMORY_CONTRACT, make_crn_schedule, summarize_paired_effects
from neda_credit_benchmark import (
    ALF_BASE_CONTRACT_VERSION,
    ALF_BASE_GATE_CONTRACT_VERSION,
    ALF_BRANCH_CONTRACT_VERSION,
    build_credit_anchors,
    deterministic_categorical_choice,
    quantile_anchor_turns,
)
from neda_data_contract import make_sampling_spec
from neda_repro import (
    model_identities_match,
    seed_everything,
    sha256_file,
    sha256_json,
    stable_seed,
)
from neda_step_token_counterfactual import (
    REPLACEMENT_CONTRACT,
    replacement_positions,
    select_action_token_coordinates,
    select_step_token_coordinates,
)
from r002_alfworld import build_action_trie


def _base_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "task_index": int(args.task_index),
        "rollouts": int(args.rollouts),
        "anchors_per_episode": int(args.anchors_per_episode),
        "max_steps": int(args.max_steps),
        "max_history": int(args.max_history),
        "thought_order": str(args.thought_order),
        "action_order": str(args.action_order),
        "action_grammar": str(args.action_grammar),
        "temperature": float(args.temperature),
        "base_rollout_seed": int(args.base_rollout_seed),
        "code_bundle_sha256": str(args.code_bundle_sha256),
    }


def run_base(args: argparse.Namespace, runtime: Runtime) -> None:
    manager, game_files, split_spec, split_root = environment_manager(args)
    game_ids = list(game_files)
    if not (0 <= int(args.task_index) < len(game_ids)):
        raise ValueError("ALFWorld credit task index is outside frozen dev split")
    game_id = game_ids[int(args.task_index)]
    protocol = _base_protocol(args)
    if os.path.isfile(args.out):
        artifact = json.load(open(args.out, "r", encoding="utf-8"))
        if (
            artifact.get("contract_version") != ALF_BASE_CONTRACT_VERSION
            or artifact.get("game_id") != game_id
            or artifact.get("protocol") != protocol
            or not model_identities_match(
                artifact.get("model_identity", {}), runtime.model_identity
            )
        ):
            raise ValueError("existing ALF credit base checkpoint drift")
        if artifact.get("complete"):
            print("[credit-alf/base] already complete: {}".format(args.out), flush=True)
            return
    else:
        artifact = {
            "contract_version": ALF_BASE_CONTRACT_VERSION,
            "artifact_kind": "stateful-credit-base",
            "complete": False,
            "memory_contract": MEMORY_CONTRACT,
            "code_bundle_sha256": str(args.code_bundle_sha256),
            "model_identity": runtime.model_identity,
            "model_identity_sha256": runtime.model_identity["identity_sha256"],
            "seed_registration": args.seed_registration,
            "split": {
                "name": args.split_name,
                "manifest": os.path.realpath(args.split_manifest),
                "game_ids_sha256": split_spec["game_ids_sha256"],
                "source_root": os.path.realpath(split_root),
            },
            "game_id": game_id,
            "protocol": protocol,
            "base_episodes": [],
            "anchors": [],
        }
        atomic_json_dump(artifact, args.out)
    completed_rollouts = {int(row["rollout_id"]) for row in artifact["base_episodes"]}
    env = make_repeated_env(manager, game_files[game_id], int(args.rollouts) + 4)
    try:
        for rollout_id in range(int(args.rollouts)):
            if rollout_id in completed_rollouts:
                continue
            episode = collect_base_episode(
                runtime,
                env,
                game_id,
                rollout_id,
                args.base_rollout_seed,
                args.max_steps,
                args.max_history,
            )
            # Fail before adding a partial episode if four identifiable turns
            # are unavailable; the fixed 16-way branch mapping depends on it.
            quantile_anchor_turns(
                len(episode["turns"]), int(args.anchors_per_episode)
            )
            artifact["base_episodes"].append(episode)
            artifact["anchors"] = build_credit_anchors(
                artifact["base_episodes"], int(args.anchors_per_episode)
            )
            atomic_json_dump(artifact, args.out)
            print(
                "[credit-alf/base] task={} rollout={}/{} H={} return={:.3f}".format(
                    args.task_index,
                    rollout_id + 1,
                    args.rollouts,
                    episode["horizon"],
                    episode["return"],
                ),
                flush=True,
            )
    finally:
        close_env(env)
    expected_anchors = int(args.rollouts) * int(args.anchors_per_episode)
    if (
        len(artifact["base_episodes"]) != int(args.rollouts)
        or len(artifact["anchors"]) != expected_anchors
    ):
        raise ValueError("ALF credit base completion count drift")
    artifact["complete"] = True
    artifact["n_episodes"] = len(artifact["base_episodes"])
    artifact["n_anchors"] = len(artifact["anchors"])
    artifact["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json_dump(artifact, args.out)
    gate = {
        "contract_version": ALF_BASE_GATE_CONTRACT_VERSION,
        "status": "PASS",
        "environment": "alfworld",
        "task_index": int(args.task_index),
        "game_id": game_id,
        "base_artifact": os.path.realpath(args.out),
        "base_artifact_sha256": sha256_file(args.out),
        "model_identity_sha256": runtime.model_identity["identity_sha256"],
        "code_bundle_sha256": str(args.code_bundle_sha256),
        "n_episodes": len(artifact["base_episodes"]),
        "n_anchors": len(artifact["anchors"]),
    }
    gate["gate_sha256"] = sha256_json(gate)
    atomic_json_dump(gate, args.gate)
    print("[credit-alf/base] PASS out={}".format(args.out), flush=True)


def load_base(args: argparse.Namespace, runtime: Runtime) -> Dict[str, Any]:
    artifact = json.load(open(args.base, "r", encoding="utf-8"))
    if (
        artifact.get("contract_version") != ALF_BASE_CONTRACT_VERSION
        or artifact.get("artifact_kind") != "stateful-credit-base"
        or not artifact.get("complete")
        or artifact.get("memory_contract") != MEMORY_CONTRACT
    ):
        raise ValueError("ALF credit base is incomplete or incompatible")
    if not model_identities_match(artifact["model_identity"], runtime.model_identity):
        raise ValueError("ALF credit base and branch checkpoint differ")
    if artifact.get("seed_registration") != args.seed_registration:
        raise ValueError("ALF credit base/branch seed registration drift")
    if artifact.get("code_bundle_sha256") != str(args.code_bundle_sha256):
        raise ValueError("ALF credit base/branch code bundle drift")
    if sha256_file(args.base) != str(args.expected_base_sha256):
        raise ValueError("ALF credit base SHA differs from registered gate")
    return artifact


def _find_reference(rows: Sequence[Mapping[str, Any]], coordinate_id: str):
    for row in rows:
        if str(row.get("coordinate_id")) == str(coordinate_id):
            return row
    return None


def _summary_pairs(
    pairs: Sequence[Mapping[str, Any]], originals: Mapping[str, Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    return [
        {
            "crn_schedule": row["crn_schedule"],
            "original": originals[str(row["sample_id"])],
            "alternative": row["alternative"],
            "paired_effect": float(row["paired_effect"]),
        }
        for row in pairs
    ]


def sample_policy_action_excluding(
    runtime: Runtime,
    action_prefix_ids: Sequence[int],
    commands: Sequence[str],
    original_action: str,
    base_seed: int,
    max_attempts: int = 128,
) -> Dict[str, Any]:
    """Sample the behavior Action distribution conditional on changing Action.

    Each proposal uses the unmodified full legal-command Trie.  Rejecting only
    the realized command therefore targets the old behavior policy conditioned
    on ``Action != original``; rebuilding a Trie with the original leaf removed
    would instead introduce prefix-local renormalization bias.  If a very sharp
    policy exhausts the registered rejection cap, enumerate the exact Trie-leaf
    probabilities in the same fixed-width learner layout and draw from the
    normalized non-realized mass.  Both paths target the identical conditional.
    """

    commands = [str(value) for value in commands]
    original_action = str(original_action)
    if original_action not in commands:
        raise ValueError("realized Action is outside the frozen admissible set")
    if len(set(commands) - {original_action}) < 1:
        raise ValueError("conditional Action intervention has no legal alternative")
    max_attempts = int(max_attempts)
    if max_attempts < 1:
        raise ValueError("conditional Action rejection cap must be positive")
    draws = []
    for attempt in range(max_attempts):
        draw_seed = stable_seed(
            int(base_seed), "old-policy-action-conditional-rejection", attempt
        )
        action = runtime.policy_action_sample(
            action_prefix_ids, commands, int(draw_seed)
        )
        draws.append(
            {"attempt": attempt, "decision_seed": int(draw_seed), "action": action}
        )
        if action != original_action:
            result = {
                "contract_version": "neda-old-policy-action-conditional-v2",
                "distribution": "behavior-policy-Action-given-not-realized-Action",
                "sampler": "exact-rejection",
                "base_seed": int(base_seed),
                "max_attempts": max_attempts,
                "original_action": original_action,
                "admissible_commands_sha256": sha256_json(sorted(commands)),
                "draws": draws,
                "accepted_action": action,
                "accepted_decision_seed": int(draw_seed),
                "n_draws": len(draws),
            }
            result["receipt_sha256"] = sha256_json(result)
            return result

    trie = build_action_trie(runtime.tokenizer, commands)
    replay_width = int(runtime.args.action_gen_length)
    sampling = make_sampling_spec(
        temperature=float(runtime.args.temperature),
        top_k=0,
        top_p=1.0,
        constraint="trie",
    )
    tokenizations: Dict[str, List[int]] = {}
    tokenization_owner: Dict[Any, str] = {}
    for command in sorted(set(commands)):
        token_ids = [
            int(value)
            for value in runtime.tokenizer(
                command, add_special_tokens=False
            )["input_ids"]
        ]
        if not token_ids or len(token_ids) > replay_width:
            raise ValueError("admissible Action is empty or exceeds replay width")
        key = tuple(token_ids)
        if key in tokenization_owner and tokenization_owner[key] != command:
            raise ValueError("distinct admissible Actions share one token sequence")
        tokenization_owner[key] = command
        tokenizations[command] = token_ids

    state_cache: Dict[Any, Dict[str, Any]] = {}
    scored_rows = []
    finite_logprobs = []
    for command in sorted(tokenizations):
        token_ids = tokenizations[command]
        reachable = bool(trie.is_complete(token_ids)) and not bool(
            trie.allowed_next(token_ids)
        )
        log_probability = 0.0
        state_hashes = []
        if reachable:
            for position, token_id in enumerate(token_ids):
                committed = tuple(token_ids[:position])
                if committed not in state_cache:
                    state_cache[committed] = score_action_token_distribution(
                        runtime,
                        action_prefix_ids,
                        committed,
                        trie,
                        replay_width,
                        sampling,
                    )
                state = state_cache[committed]
                key = str(token_id)
                if key not in state["allowed_token_logprobs"]:
                    raise ValueError("admissible Action left the restored Trie")
                log_probability += float(state["allowed_token_logprobs"][key])
                state_hashes.append(state["state_score_sha256"])
            finite_logprobs.append(log_probability)
        scored_rows.append(
            {
                "action": command,
                "reachable_leaf": reachable,
                "token_ids_sha256": sha256_json(token_ids),
                "n_tokens": len(token_ids),
                "log_probability": log_probability if reachable else None,
                "state_score_hashes": state_hashes,
            }
        )
    if not finite_logprobs:
        raise ValueError("old policy has no reachable admissible Action leaf")
    maximum = max(finite_logprobs)
    scaled_total = math.fsum(math.exp(value - maximum) for value in finite_logprobs)
    log_support_mass = maximum + math.log(scaled_total)
    if not math.isfinite(log_support_mass) or abs(log_support_mass) > 0.02:
        raise ValueError(
            "enumerated old-policy Action mass drift: log_mass={:.6f}".format(
                log_support_mass
            )
        )
    for row in scored_rows:
        if row["reachable_leaf"]:
            row["behavior_probability"] = math.exp(
                float(row["log_probability"]) - log_support_mass
            )
        else:
            row["behavior_probability"] = 0.0
    original_row = next(
        row for row in scored_rows if row["action"] == original_action
    )
    if not original_row["reachable_leaf"]:
        raise ValueError("realized Action is not a reachable Trie leaf")
    alternative_mass = math.fsum(
        float(row["behavior_probability"])
        for row in scored_rows
        if row["action"] != original_action
    )
    if not math.isfinite(alternative_mass) or alternative_mass <= 0.0:
        raise ValueError("old policy has no finite non-realized Action mass")
    conditional = {}
    for row in scored_rows:
        probability = (
            0.0
            if row["action"] == original_action
            else float(row["behavior_probability"]) / alternative_mass
        )
        row["conditional_probability"] = probability
        if probability > 0.0:
            conditional[str(row["action"])] = probability
    enumeration_seed = stable_seed(
        int(base_seed), "old-policy-action-conditional-enumeration-v2"
    )
    categorical = deterministic_categorical_choice(conditional, enumeration_seed)
    accepted = str(categorical["selected_action"])
    enumeration = {
        "contract_version": "neda-old-policy-action-enumeration-v1",
        "decision_seed": int(enumeration_seed),
        "scoring_layout": "full-duplicate-ar-v1",
        "replay_width": replay_width,
        "sampling": sampling,
        "n_scored_states": len(state_cache),
        "log_support_mass": log_support_mass,
        "alternative_mass": alternative_mass,
        "action_table": scored_rows,
        "categorical_draw": categorical,
    }
    enumeration["enumeration_sha256"] = sha256_json(enumeration)
    result = {
        "contract_version": "neda-old-policy-action-conditional-v2",
        "distribution": "behavior-policy-Action-given-not-realized-Action",
        "sampler": "exact-enumeration-after-registered-rejection-cap",
        "base_seed": int(base_seed),
        "max_attempts": max_attempts,
        "original_action": original_action,
        "admissible_commands_sha256": sha256_json(sorted(commands)),
        "draws": draws,
        "enumeration": enumeration,
        "accepted_action": accepted,
        "accepted_decision_seed": int(enumeration_seed),
        "n_draws": len(draws),
    }
    result["receipt_sha256"] = sha256_json(result)
    return result


def run_branch(args: argparse.Namespace, runtime: Runtime) -> None:
    base = load_base(args, runtime)
    manager, game_files, split_spec, _ = environment_manager(args)
    if split_spec["game_ids_sha256"] != base["split"]["game_ids_sha256"]:
        raise ValueError("ALF credit branch split drift")
    if not (0 <= int(args.anchor_index) < len(base["anchors"])):
        raise ValueError("ALF credit anchor index is outside the fixed base")
    anchor = base["anchors"][int(args.anchor_index)]
    episodes = {row["episode_id"]: row for row in base["base_episodes"]}
    episode = episodes[anchor["episode_id"]]
    turn = episode["turns"][int(anchor["turn_id"])]
    thought_trace = turn["decision_traces"]["thought"]
    action_trace = turn["decision_traces"]["action"]
    commands = list(turn["state_before"]["admissible_commands"])
    thought_selection = select_step_token_coordinates(
        thought_trace,
        anchor["anchor_id"],
        args.selection_seed,
        block_size=args.block_length,
        n_steps=2,
        tokens_per_step=2,
    )
    trie = build_action_trie(runtime.tokenizer, commands)
    if not trie.is_complete(action_trace["response_ids"]):
        raise ValueError("base Action trace is outside the frozen Trie")
    action_selection = select_action_token_coordinates(
        action_trace,
        anchor["anchor_id"],
        args.selection_seed,
        trie.allowed_next,
        max_action_tokens=2,
        min_action_tokens=1,
    )
    selection = {"thought": thought_selection, "action": action_selection}
    selection["selection_sha256"] = sha256_json(selection)
    coordinates = replacement_positions(thought_selection) + list(
        action_selection["coordinates"]
    )
    protocol = {
        "k": int(args.k),
        "branch_seed": int(args.branch_seed),
        "selection_seed": int(args.selection_seed),
        "n_thought_steps": 2,
        "thought_tokens_per_step": 2,
        "min_action_tokens": 1,
        "max_action_tokens": 2,
        "replacement_contract": REPLACEMENT_CONTRACT,
        "turn_intervention": "old-policy-Action-conditional-hybrid-exact-v2",
        "turn_intervention_max_attempts": 128,
        "local_intervention": REPLACEMENT_CONTRACT,
        "max_steps": int(base["protocol"]["max_steps"]),
        "max_history": int(base["protocol"]["max_history"]),
        "logprob_tolerance": float(args.logprob_tolerance),
        "code_bundle_sha256": str(args.code_bundle_sha256),
    }
    if os.path.isfile(args.out):
        artifact = json.load(open(args.out, "r", encoding="utf-8"))
        if (
            artifact.get("contract_version") != ALF_BRANCH_CONTRACT_VERSION
            or artifact.get("base_artifact_sha256") != str(args.expected_base_sha256)
            or artifact.get("anchor") != anchor
            or artifact.get("protocol") != protocol
            or artifact.get("selection") != selection
        ):
            raise ValueError("existing ALF credit branch checkpoint drift")
        if artifact.get("complete"):
            print("[credit-alf/branch] already complete: {}".format(args.out), flush=True)
            return
    else:
        artifact = {
            "contract_version": ALF_BRANCH_CONTRACT_VERSION,
            "artifact_kind": "counterfactual-credit-reference",
            "complete": False,
            "environment": "alfworld",
            "memory_contract": MEMORY_CONTRACT,
            "code_bundle_sha256": str(args.code_bundle_sha256),
            "model_identity_sha256": runtime.model_identity["identity_sha256"],
            "seed_registration": args.seed_registration,
            "base_artifact": os.path.realpath(args.base),
            "base_artifact_sha256": str(args.expected_base_sha256),
            "task_index": int(args.task_index),
            "anchor_index": int(args.anchor_index),
            "anchor": anchor,
            "protocol": protocol,
            "selection": selection,
            "originals": {},
            "turn_pairs": [],
            "local_references": [],
        }
        atomic_json_dump(artifact, args.out)
    resets = int(args.k) * (2 + len(coordinates)) + 8
    env = make_repeated_env(manager, game_files[anchor["game_id"]], resets)
    try:
        schedules = {}
        for sample_id in range(int(args.k)):
            schedule = make_crn_schedule(
                args.branch_seed,
                anchor["anchor_id"],
                0,
                sample_id,
                max(0, int(protocol["max_steps"]) - int(anchor["turn_id"]) - 1),
            )
            schedules[sample_id] = schedule
            key = str(sample_id)
            if key not in artifact["originals"]:
                artifact["originals"][key] = run_one_branch(
                    runtime,
                    env,
                    episode,
                    anchor,
                    schedule,
                    "original",
                    {},
                    protocol["max_steps"],
                    protocol["max_history"],
                )
                atomic_json_dump(artifact, args.out)
        existing_turn = {int(row["sample_id"]) for row in artifact["turn_pairs"]}
        for sample_id in range(int(args.k)):
            if sample_id in existing_turn:
                continue
            schedule = schedules[sample_id]
            action_seed = stable_seed(
                schedule["intervention_seed"], "turn-action-excluding-original"
            )
            conditional = sample_policy_action_excluding(
                runtime,
                action_trace["prefix_ids"],
                commands,
                turn["executed_action"],
                action_seed,
                max_attempts=protocol["turn_intervention_max_attempts"],
            )
            action = conditional["accepted_action"]
            regenerated = {
                "action": action,
                "generation": replace_action_in_generation(turn["generation"], action),
                "policy_decision_seed": conditional["accepted_decision_seed"],
                "conditional_sampling": conditional,
            }
            alternative = run_one_branch(
                runtime,
                env,
                episode,
                anchor,
                schedule,
                "alternative",
                {"kind": "explicit_counterfactual_action", **regenerated},
                protocol["max_steps"],
                protocol["max_history"],
            )
            original = artifact["originals"][str(sample_id)]
            artifact["turn_pairs"].append(
                {
                    "sample_id": sample_id,
                    "crn_schedule": schedule,
                    "original_key": str(sample_id),
                    "alternative_action": regenerated,
                    "alternative": alternative,
                    "paired_effect": float(original["return"])
                    - float(alternative["return"]),
                }
            )
            atomic_json_dump(artifact, args.out)
        for coordinate in coordinates:
            if _find_reference(
                artifact["local_references"], coordinate["coordinate_id"]
            ) is not None:
                continue
            pairs = []
            for sample_id in range(int(args.k)):
                schedule = schedules[sample_id]
                replacement_seed = stable_seed(
                    schedule["intervention_seed"],
                    coordinate["coordinate_id"],
                    "{}-replacement".format(coordinate["level"]),
                )
                if coordinate["level"] == "action_token":
                    action_result = resample_recorded_action_coordinate(
                        runtime,
                        action_trace,
                        coordinate,
                        commands,
                        replacement_seed,
                        args.logprob_tolerance,
                        turn["generation"],
                    )
                    replacement = action_result["replacement"]
                    regenerated = action_result["regenerated_action"]
                else:
                    replacement = resample_recorded_thought_coordinate(
                        runtime,
                        thought_trace,
                        coordinate,
                        replacement_seed,
                        args.block_length,
                        args.gen_length,
                        args.logprob_tolerance,
                    )
                    action_seed = stable_seed(
                        schedule["intervention_seed"],
                        coordinate["coordinate_id"],
                        "regenerated-action",
                    )
                    regenerated = action_from_modified_thought(
                        runtime,
                        thought_trace,
                        replacement["modified_response_ids"],
                        commands,
                        action_seed,
                    )
                alternative = run_one_branch(
                    runtime,
                    env,
                    episode,
                    anchor,
                    schedule,
                    "alternative",
                    {"kind": "explicit_counterfactual_action", **regenerated},
                    protocol["max_steps"],
                    protocol["max_history"],
                )
                original = artifact["originals"][str(sample_id)]
                pairs.append(
                    {
                        "sample_id": sample_id,
                        "crn_schedule": schedule,
                        "original_key": str(sample_id),
                        "replacement": replacement,
                        "regenerated_action": regenerated,
                        "action_changed": regenerated["action"]
                        != turn["executed_action"],
                        "alternative": alternative,
                        "paired_effect": float(original["return"])
                        - float(alternative["return"]),
                    }
                )
            artifact["local_references"].append(
                {
                    **coordinate,
                    "pairs": pairs,
                    "reference": summarize_paired_effects(
                        _summary_pairs(pairs, artifact["originals"])
                    ),
                }
            )
            atomic_json_dump(artifact, args.out)
            print(
                "[credit-alf/branch] task={} anchor={} level={} refs={}/{}".format(
                    args.task_index,
                    args.anchor_index,
                    coordinate["level"],
                    len(artifact["local_references"]),
                    len(coordinates),
                ),
                flush=True,
            )
    finally:
        close_env(env)
    if set(artifact["originals"]) != {str(index) for index in range(int(args.k))}:
        raise ValueError("shared original completion drift")
    if len(artifact["turn_pairs"]) != int(args.k):
        raise ValueError("turn reference completion drift")
    if len(artifact["local_references"]) != len(coordinates):
        raise ValueError("local reference completion drift")
    if any(len(row["pairs"]) != int(args.k) for row in artifact["local_references"]):
        raise ValueError("local reference K completion drift")
    artifact["turn_reference"] = summarize_paired_effects(
        _summary_pairs(artifact["turn_pairs"], artifact["originals"])
    )
    artifact["complete"] = True
    artifact["n_local_references"] = len(artifact["local_references"])
    artifact["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json_dump(artifact, args.out)
    print("[credit-alf/branch] PASS out={}".format(args.out), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--mode", choices=("base", "branch"), required=True)
    result.add_argument("--model_dir", required=True)
    result.add_argument("--alfworld_config", required=True)
    result.add_argument("--prompt_json", required=True)
    result.add_argument("--split_manifest", required=True)
    result.add_argument("--counterfactual_seed_manifest", required=True)
    result.add_argument("--split_name", default="dev_seen")
    result.add_argument("--split_root")
    result.add_argument("--seed_phase", default="submission")
    result.add_argument("--seed_replicate", type=int, default=0)
    result.add_argument("--task_index", type=int, required=True)
    result.add_argument("--rollouts", type=int, default=4)
    result.add_argument("--anchors_per_episode", type=int, default=4)
    result.add_argument("--anchor_index", type=int, default=0)
    result.add_argument("--base")
    result.add_argument("--expected_base_sha256", default="")
    result.add_argument("--out", required=True)
    result.add_argument("--gate")
    result.add_argument("--thought_order", default="ao")
    result.add_argument("--action_order", default="ar")
    result.add_argument("--action_grammar", default="trie")
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--gen_length", type=int, default=128)
    result.add_argument("--action_gen_length", type=int, default=24)
    result.add_argument("--block_length", type=int, default=4)
    result.add_argument("--denoising_steps", type=int, default=4)
    result.add_argument("--max_history", type=int, default=24)
    result.add_argument("--max_steps", type=int, default=30)
    result.add_argument("--num_games", type=int, default=64)
    result.add_argument("--rollouts_per_game", type=int, default=4)
    result.add_argument("--game_offset", type=int, default=0)
    # These defaults are the frozen submission replicate-0 tuple in
    # manifests/neda_v4_counterfactual_seeds.json.  Payloads still pass them
    # explicitly so a manifest/CLI drift fails in validate_registered_seeds.
    result.add_argument("--base_rollout_seed", type=int, default=62001)
    result.add_argument("--branch_seed", type=int, default=72001)
    result.add_argument("--selection_seed", type=int, default=82001)
    result.add_argument("--k", type=int, default=8)
    result.add_argument("--logprob_tolerance", type=float, default=0.05)
    result.add_argument("--code_bundle_sha256", default="")
    return result


def main() -> None:
    args = parser().parse_args()
    if (
        args.thought_order != "ao"
        or args.action_order != "ar"
        or args.action_grammar != "trie"
        or args.block_length != 4
        or args.denoising_steps != 4
    ):
        raise ValueError("credit benchmark requires canonical AO-Thought/AR+Trie-Action")
    if args.mode == "base" and not args.gate:
        raise ValueError("base mode requires --gate")
    if args.mode == "branch" and (not args.base or not args.expected_base_sha256):
        raise ValueError("branch mode requires frozen base path and SHA")
    if len(str(args.code_bundle_sha256)) != 64:
        raise ValueError("credit benchmark requires a frozen 64-hex code bundle SHA")
    args.seed_registration = validate_registered_seeds(args)
    runtime = Runtime(args)
    if args.mode == "base":
        run_base(args, runtime)
    else:
        run_branch(args, runtime)


if __name__ == "__main__":
    main()
