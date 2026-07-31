#!/usr/bin/env python3
"""ALFWorld V4-O02/O03 controlled Thought step/token interventions."""

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from alfworld_counterfactual import (
    Runtime,
    admissible,
    atomic_json_dump,
    close_env,
    environment_manager,
    load_base_artifact,
    make_repeated_env,
    replace_action_in_generation,
    run_one_branch,
    validate_registered_seeds,
)
from neda_counterfactual import (
    MEMORY_CONTRACT,
    make_crn_schedule,
    select_retest_anchor_ids,
    summarize_paired_effects,
)
from neda_data_contract import (
    EXACT_ACTION_SCORING_LAYOUT,
    build_structural_action_prefix,
    trim_generation_trace,
)
from neda_repro import seed_everything, sha256_file, sha256_json, stable_seed
from neda_step_token_counterfactual import (
    REPLACEMENT_CONTRACT,
    STEP_TOKEN_CONTRACT_VERSION,
    replacement_positions,
    select_action_token_coordinates,
    select_step_token_coordinates,
)
from neda_torch_replay import exact_replay_numerics, make_basic_block_attention
from r002_alfworld import build_action_trie, exact_ar_action_generate


ARTIFACT_KIND = "step-token-branch-results"


def transformed_log_probs(logits: torch.Tensor, sampling: Mapping[str, Any]) -> torch.Tensor:
    """Apply the exact rollout sampling transforms without drawing a token."""

    logits = logits.float()
    temperature = float(sampling.get("temperature", 1.0))
    top_k = int(sampling.get("top_k", 0))
    top_p = float(sampling.get("top_p", 1.0))
    if temperature <= 0:
        raise ValueError("sampling temperature must be positive")
    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        logits = torch.where(
            logits < values[..., -1, None],
            torch.full_like(logits, float("-inf")),
            logits,
        )
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        removed = cumulative > top_p
        removed[..., 1:] = removed[..., :-1].clone()
        removed[..., 0] = False
        mask = torch.scatter(
            torch.full_like(logits, False, dtype=torch.bool),
            -1,
            sorted_indices,
            removed,
        )
        logits = logits.masked_fill(mask, float("-inf"))
    return F.log_softmax(logits, dim=-1)


@torch.no_grad()
def resample_recorded_thought_coordinate(
    runtime: Runtime,
    trace: Mapping[str, Any],
    coordinate: Mapping[str, Any],
    replacement_seed: int,
    block_size: int,
    gen_length: int,
    logprob_tolerance: float,
) -> Dict[str, Any]:
    """Resample selected commits from their exact recorded KV-cache state."""

    prefix = [int(value) for value in trace["prefix_ids"]]
    response = [int(value) for value in trace["response_ids"]]
    step_map = [int(value) for value in trace["step_map"]]
    positions = [int(value) for value in coordinate["positions"]]
    step_id = int(coordinate["step_id"])
    if not positions or any(step_map[position] != step_id for position in positions):
        raise ValueError("replacement positions do not match the recorded step")
    absolute_positions = [len(prefix) + position for position in positions]
    target_blocks = {position // int(block_size) for position in absolute_positions}
    if len(target_blocks) != 1:
        raise ValueError("one coordinate crossed decoder blocks")
    target_block = next(iter(target_blocks))
    recorded_end = len(prefix) + len(response)
    if (target_block + 1) * int(block_size) > recorded_end:
        raise ValueError("target block is not fully recorded")

    num_blocks = (len(prefix) + int(gen_length) + int(block_size) - 1) // int(block_size)
    total = num_blocks * int(block_size)
    if recorded_end > total:
        raise ValueError("recorded Thought exceeds configured generation width")
    device = runtime.model.device
    x = torch.full(
        (1, total), int(runtime.mask_id), dtype=torch.long, device=device
    )
    x[:, : len(prefix)] = torch.as_tensor(prefix, dtype=torch.long, device=device)
    x[:, len(prefix) : recorded_end] = torch.as_tensor(
        response, dtype=torch.long, device=device
    )
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    attention = (
        block_mask.repeat_interleave(int(block_size), 0)
        .repeat_interleave(int(block_size), 1)
        .unsqueeze(0)
        .unsqueeze(1)
    )
    position_ids = torch.arange(total, device=device).unsqueeze(0)
    cache = DynamicCache()
    prefill_blocks = len(prefix) // int(block_size)
    prefill_length = prefill_blocks * int(block_size)
    if prefill_length:
        runtime.model(
            x[:, :prefill_length],
            attention_mask=attention[:, :, :prefill_length, :prefill_length],
            position_ids=position_ids[:, :prefill_length],
            past_key_values=cache,
            use_cache=True,
            store_kv=True,
        )
    for block_index in range(prefill_blocks, target_block):
        left = block_index * int(block_size)
        right = left + int(block_size)
        block = x[:, left:right].clone()
        if bool((block == int(runtime.mask_id)).any()):
            raise ValueError("an earlier recorded block is incomplete")
        runtime.model(
            block,
            attention_mask=attention[:, :, left:right, :right],
            position_ids=position_ids[:, left:right],
            past_key_values=cache,
            use_cache=True,
            store_kv=True,
        )

    left = target_block * int(block_size)
    right = left + int(block_size)
    current = x[:, left:right].clone()
    for response_position, recorded_step in enumerate(step_map):
        absolute = len(prefix) + response_position
        if left <= absolute < right and recorded_step >= step_id:
            current[:, absolute - left] = int(runtime.mask_id)
    output = runtime.model(
        current,
        attention_mask=attention[:, :, left:right, :right],
        position_ids=position_ids[:, left:right],
        past_key_values=cache,
        use_cache=True,
        store_kv=False,
    )
    local_positions = [value - left for value in absolute_positions]
    selected_logits = output.logits[:, local_positions, :][0]
    sampling = dict(trace.get("sampling", {}))
    if sampling.get("constraint", "none") != "none":
        raise ValueError("Thought replacement requires an unconstrained trace")
    log_probs = transformed_log_probs(selected_logits, sampling)
    original_ids = torch.as_tensor(
        [response[position] for position in positions],
        dtype=torch.long,
        device=device,
    )
    rescored = log_probs.gather(-1, original_ids[:, None]).squeeze(-1)
    stored = torch.as_tensor(
        [float(trace["behavior_logprobs"][position]) for position in positions],
        dtype=torch.float32,
        device=device,
    )
    errors = (rescored.float() - stored).abs()
    max_error = float(errors.max().item())
    if not math.isfinite(max_error) or max_error > float(logprob_tolerance):
        raise ValueError(
            "recorded Thought state logprob mismatch {:.6f} > {:.6f}".format(
                max_error, float(logprob_tolerance)
            )
        )

    conditioned = log_probs.clone()
    conditioned.scatter_(-1, original_ids[:, None], float("-inf"))
    normalizer = torch.logsumexp(conditioned, dim=-1, keepdim=True)
    if not bool(torch.isfinite(normalizer).all()):
        raise ValueError("old-policy replacement has no non-realized support")
    conditioned = conditioned - normalizer
    seed_everything(int(replacement_seed))
    sampled = torch.multinomial(conditioned.exp(), num_samples=1).squeeze(-1)
    sampled_logprobs = conditioned.gather(-1, sampled[:, None]).squeeze(-1)
    if bool((sampled == original_ids).any()):
        raise ValueError("conditioned replacement reproduced an excluded token")
    modified = list(response)
    for position, token_id in zip(positions, sampled.tolist()):
        modified[position] = int(token_id)
    result = {
        "contract": REPLACEMENT_CONTRACT,
        "coordinate_id": str(coordinate["coordinate_id"]),
        "level": str(coordinate["level"]),
        "step_id": step_id,
        "positions": positions,
        "replacement_seed": int(replacement_seed),
        "state_sha256": sha256_json(
            {
                "trace_sha256": trace.get("trace_sha256"),
                "step_id": step_id,
                "committed_before": [
                    response[index] if step_map[index] < step_id else None
                    for index in range(len(response))
                ],
            }
        ),
        "original_token_ids": original_ids.tolist(),
        "replacement_token_ids": sampled.tolist(),
        "stored_behavior_logprobs": stored.tolist(),
        "rescored_behavior_logprobs": rescored.float().tolist(),
        "replacement_logprobs": sampled_logprobs.float().tolist(),
        "max_abs_behavior_logprob_error": max_error,
        "modified_response_ids": modified,
    }
    result["replacement_sha256"] = sha256_json(result)
    return result


def action_from_modified_thought(
    runtime: Runtime,
    trace: Mapping[str, Any],
    modified_response_ids: Sequence[int],
    commands: Sequence[str],
    action_seed: int,
) -> Dict[str, Any]:
    thought_ids = [int(value) for value in modified_response_ids]
    thought_text = runtime.tokenizer.decode(thought_ids, skip_special_tokens=True)
    structural = build_structural_action_prefix(
        runtime.tokenizer,
        [int(value) for value in trace["prefix_ids"]],
        thought_text,
        thought_ids,
    )
    effective_thought_ids = thought_ids[: int(structural["thought_end"])]
    effective_thought = runtime.tokenizer.decode(
        effective_thought_ids, skip_special_tokens=True
    )
    action = runtime.policy_action_sample(
        structural["action_prefix_ids"], commands, int(action_seed)
    )
    generation = "{}{}Action: {}".format(
        effective_thought.rstrip(), "\n" if effective_thought.strip() else "", action
    )
    return {
        "action": action,
        "generation": generation,
        "policy_decision_seed": int(action_seed),
        "thought_text": effective_thought,
        "thought_ids": effective_thought_ids,
        "structural_action_prefix_sha256": sha256_json(
            structural["action_prefix_ids"]
        ),
        "marker_generated_after_replacement": bool(structural["marker_generated"]),
    }


@torch.no_grad()
def score_action_token_distribution(
    runtime: Runtime,
    action_prefix_ids: Sequence[int],
    committed: Sequence[int],
    trie: Any,
    replay_width: int,
    sampling: Mapping[str, Any],
) -> Dict[str, Any]:
    """Score every legal next token in the canonical AR learner layout."""

    prefix = [int(value) for value in action_prefix_ids]
    committed = [int(value) for value in committed]
    replay_width = int(replay_width)
    if replay_width <= 0 or len(committed) >= replay_width:
        raise ValueError("Action state is outside the fixed replay width")
    if sampling.get("constraint") != "trie":
        raise ValueError("Action state scoring requires the frozen Trie distribution")
    allowed = {int(value) for value in trie.allowed_next(committed)}
    if not allowed:
        raise ValueError("Action state has no legal continuation")

    device = runtime.model.device
    sequence = torch.full(
        (1, len(prefix) + replay_width),
        int(runtime.mask_id),
        dtype=torch.long,
        device=device,
    )
    sequence[:, : len(prefix)] = torch.as_tensor(
        prefix, dtype=torch.long, device=device
    )
    if committed:
        sequence[:, len(prefix) : len(prefix) + len(committed)] = torch.as_tensor(
            committed, dtype=torch.long, device=device
        )
    extended = torch.cat([sequence, sequence[:, len(prefix) :]], dim=1)
    total = len(prefix) + 2 * replay_width
    attention = make_basic_block_attention(
        total, len(prefix), block_size=1, device=device
    )
    base_positions = torch.arange(
        len(prefix) + replay_width, dtype=torch.long, device=device
    ).unsqueeze(0)
    position_ids = torch.cat(
        [base_positions, base_positions[:, len(prefix) :]], dim=1
    )
    with exact_replay_numerics():
        logits = runtime.model(
            input_ids=extended,
            attention_mask=attention,
            position_ids=position_ids,
            use_cache=False,
            store_kv=False,
        ).logits
    query = len(prefix) + replay_width + len(committed)
    scores = logits[:, query, :].float()
    vocabulary_mask = torch.full_like(scores, float("-inf"))
    vocabulary_mask[:, torch.as_tensor(sorted(allowed), device=device)] = 0.0
    log_probs = transformed_log_probs(scores + vocabulary_mask, sampling)
    result = {
        "committed_token_ids": committed,
        "allowed_token_logprobs": {
            str(token_id): float(log_probs[0, token_id].item())
            for token_id in sorted(allowed)
        },
        "allowed_token_ids_sha256": sha256_json(sorted(allowed)),
        "n_allowed_token_ids": len(allowed),
        "scoring_layout": EXACT_ACTION_SCORING_LAYOUT,
        "replay_width": replay_width,
    }
    result["state_score_sha256"] = sha256_json(result)
    return result


@torch.no_grad()
def score_recorded_action_token(
    runtime: Runtime,
    trace: Mapping[str, Any],
    trie: Any,
    position: int,
) -> Dict[str, Any]:
    """Re-score one realized AR+Trie Action token in rollout layout."""

    prefix = [int(value) for value in trace["prefix_ids"]]
    response = [int(value) for value in trace["response_ids"]]
    position = int(position)
    if not (0 <= position < len(response)):
        raise ValueError("Action-token position is outside the recorded response")
    if trace.get("scoring_layout") != EXACT_ACTION_SCORING_LAYOUT:
        raise ValueError("Action trace is not in the exact duplicated AR layout")
    width = int(trace.get("replay_width", 0))
    if width < len(response):
        raise ValueError("Action trace replay width does not cover its response")
    sampling = dict(trace.get("sampling", {}))
    if sampling.get("constraint") != "trie":
        raise ValueError("Action-token reference requires a stored Trie distribution")
    committed = response[:position]
    allowed = {int(value) for value in trie.allowed_next(committed)}
    realized = int(response[position])
    if realized not in allowed:
        raise ValueError("recorded Action token is outside the restored Trie state")

    state_score = score_action_token_distribution(
        runtime,
        prefix,
        committed,
        trie,
        width,
        sampling,
    )
    rescored = float(state_score["allowed_token_logprobs"][str(realized)])
    stored = float(trace["behavior_logprobs"][position])
    return {
        "position": position,
        "realized_token_id": realized,
        "stored_behavior_logprob": stored,
        "rescored_behavior_logprob": rescored,
        "abs_behavior_logprob_error": abs(rescored - stored),
        "allowed_token_ids_sha256": state_score["allowed_token_ids_sha256"],
        "n_allowed_token_ids": state_score["n_allowed_token_ids"],
        "state_score_sha256": state_score["state_score_sha256"],
    }


@torch.no_grad()
def resample_recorded_action_coordinate(
    runtime: Runtime,
    trace: Mapping[str, Any],
    coordinate: Mapping[str, Any],
    commands: Sequence[str],
    replacement_seed: int,
    logprob_tolerance: float,
    base_generation: str,
) -> Dict[str, Any]:
    """Intervene on one realized Action token and resample a legal suffix."""

    prefix = [int(value) for value in trace["prefix_ids"]]
    response = [int(value) for value in trace["response_ids"]]
    positions = [int(value) for value in coordinate["positions"]]
    if len(positions) != 1:
        raise ValueError("Action-token intervention requires exactly one position")
    position = positions[0]
    trie = build_action_trie(runtime.tokenizer, list(commands))
    if not trie.is_complete(response):
        raise ValueError("recorded Action response is not a complete admissible command")
    score = score_recorded_action_token(runtime, trace, trie, position)
    max_error = float(score["abs_behavior_logprob_error"])
    if not math.isfinite(max_error) or max_error > float(logprob_tolerance):
        raise ValueError(
            "recorded Action state logprob mismatch {:.6f} > {:.6f}".format(
                max_error, float(logprob_tolerance)
            )
        )
    realized = int(response[position])

    def intervention_constraint(committed: Sequence[int]):
        committed = [int(value) for value in committed]
        offset = len(committed)
        allowed = {int(value) for value in trie.allowed_next(committed)}
        if offset < position:
            expected = int(response[offset])
            if expected not in allowed:
                raise ValueError("restored Action prefix left the admissible Trie")
            return {expected}
        if offset == position:
            alternatives = allowed - {realized}
            if not alternatives:
                raise ValueError("selected Action token has no non-realized support")
            return alternatives
        return allowed

    seed_everything(int(replacement_seed))
    tensor = torch.as_tensor([prefix], dtype=torch.long, device=runtime.model.device)
    output, raw_trace = exact_ar_action_generate(
        runtime.model,
        tensor,
        runtime.mask_id,
        gen_length=int(trace["replay_width"]),
        temperature=float(trace["sampling"]["temperature"]),
        top_k=int(trace["sampling"].get("top_k", 0)),
        top_p=float(trace["sampling"].get("top_p", 1.0)),
        stop_ids=runtime.stop_ids,
        constraint=intervention_constraint,
        constraint_name="trie-action-token-do-v1",
    )
    trimmed = trim_generation_trace(
        output[len(prefix) :].tolist(),
        raw_trace["step_map"].tolist(),
        raw_trace["behavior_logprobs"].tolist(),
        runtime.mask_id,
        runtime.stop_ids,
        confidence=raw_trace["commit_confidence"].tolist(),
        sampling=raw_trace["sampling"],
    )
    modified = [int(value) for value in trimmed["response_ids"]]
    if len(modified) <= position or modified[:position] != response[:position]:
        raise ValueError("Action intervention did not preserve the recorded prefix")
    if modified[position] == realized:
        raise ValueError("conditioned Action intervention reproduced excluded token")
    if not trie.is_complete(modified):
        raise ValueError("Action intervention did not terminate at a legal command")
    action = runtime.tokenizer.decode(modified, skip_special_tokens=True).strip()
    if action.endswith("."):
        action = action[:-1].strip()
    if action not in commands:
        raise ValueError("Action-token intervention decoded outside admissible commands")
    generation = replace_action_in_generation(str(base_generation), action)
    replacement_logprob = float(trimmed["behavior_logprobs"][position])
    replacement = {
        "contract": REPLACEMENT_CONTRACT,
        "coordinate_id": str(coordinate["coordinate_id"]),
        "level": "action_token",
        "step_id": position,
        "positions": [position],
        "replacement_seed": int(replacement_seed),
        "state_sha256": sha256_json(
            {
                "trace_sha256": trace.get("trace_sha256"),
                "position": position,
                "committed_before": response[:position],
                "admissible_commands": sorted(str(value) for value in commands),
            }
        ),
        "original_token_ids": [realized],
        "replacement_token_ids": [int(modified[position])],
        "stored_behavior_logprobs": [score["stored_behavior_logprob"]],
        "rescored_behavior_logprobs": [score["rescored_behavior_logprob"]],
        "replacement_logprobs": [replacement_logprob],
        "max_abs_behavior_logprob_error": max_error,
        "modified_response_ids": modified,
        "modified_action": action,
        "original_action": runtime.tokenizer.decode(
            response, skip_special_tokens=True
        ).strip(),
        "scoring_layout": trace["scoring_layout"],
        "replay_width": int(trace["replay_width"]),
        "allowed_state": score,
    }
    replacement["replacement_sha256"] = sha256_json(replacement)
    regenerated = {
        "action": action,
        "generation": generation,
        "policy_decision_seed": int(replacement_seed),
        "thought_text": None,
        "thought_ids": None,
        "structural_action_prefix_sha256": sha256_json(prefix),
        "marker_generated_after_replacement": False,
    }
    return {"replacement": replacement, "regenerated_action": regenerated}


def find_anchor_row(artifact: Mapping[str, Any], anchor_id: str) -> Any:
    for row in artifact.get("anchors", []):
        if row.get("anchor_id") == anchor_id:
            return row
    return None


def find_reference(repeat: Mapping[str, Any], coordinate_id: str) -> Any:
    for row in repeat.get("references", []):
        if row.get("coordinate_id") == coordinate_id:
            return row
    return None


def run(args: argparse.Namespace, runtime: Runtime) -> None:
    base = load_base_artifact(args.base_artifact, runtime)
    manager, game_files, split_spec, _ = environment_manager(args)
    if split_spec["game_ids_sha256"] != base["split"]["game_ids_sha256"]:
        raise ValueError("runtime split differs from O01 base artifact")
    game_ids = list(base["split"]["game_ids"])
    if not (0 <= int(args.game_index) < len(game_ids)):
        raise ValueError("game_index is outside the frozen O01 slice")
    game_id = game_ids[int(args.game_index)]
    episode_by_id = {row["episode_id"]: row for row in base["base_episodes"]}
    local_anchors = [row for row in base["anchors"] if row["game_id"] == game_id]
    if len(local_anchors) != int(base["protocol"]["rollouts_per_game"]) * 3:
        raise ValueError("O02 shard does not contain exactly three anchors per rollout")
    global_retest = set(
        select_retest_anchor_ids(
            [row["anchor_id"] for row in base["anchors"]],
            args.retest_fraction,
            args.selection_seed,
        )
    )
    base_sha = sha256_file(args.base_artifact)
    protocol = {
        "k": int(args.k),
        "n_steps": int(args.n_steps),
        "tokens_per_step": int(args.tokens_per_step),
        "max_action_tokens": int(args.max_action_tokens),
        "min_action_tokens": int(args.min_action_tokens),
        "block_size": int(args.block_size),
        "gen_length": int(args.gen_length),
        "retest_fraction": float(args.retest_fraction),
        "branch_seed": int(args.branch_seed),
        "selection_seed": int(args.selection_seed),
        "max_steps": int(base["protocol"]["max_steps"]),
        "max_history": int(base["protocol"]["max_history"]),
        "replacement_contract": REPLACEMENT_CONTRACT,
        "estimand": (
            "controlled old-policy replacement at a recorded Thought commit or "
            "canonical AR+Trie Action token; non-target realized prefix coordinates "
            "fixed; legal Action suffix and continuation regenerated"
        ),
        "shared_original_note": (
            "All coordinates within anchor/repeat/sample share one original branch "
            "and identical post-anchor CRN decision seeds."
        ),
        "logprob_tolerance": float(args.logprob_tolerance),
    }
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as handle:
            artifact = json.load(handle)
        if (
            artifact.get("contract_version") != STEP_TOKEN_CONTRACT_VERSION
            or artifact.get("artifact_kind") != ARTIFACT_KIND
            or artifact.get("base_artifact_sha256") != base_sha
            or artifact.get("game_id") != game_id
            or artifact.get("protocol") != protocol
        ):
            raise ValueError("existing O02 checkpoint does not match this invocation")
        if artifact.get("complete"):
            print("[O02] already complete: {}".format(args.out), flush=True)
            return
    else:
        artifact = {
            "contract_version": STEP_TOKEN_CONTRACT_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "phase": "o02_o03",
            "complete": False,
            "memory_contract": MEMORY_CONTRACT,
            "model_identity_sha256": runtime.model_identity["identity_sha256"],
            "base_artifact": os.path.realpath(args.base_artifact),
            "base_artifact_sha256": base_sha,
            "seed_registration": args.seed_registration,
            "game_index": int(args.game_index),
            "game_id": game_id,
            "protocol": protocol,
            "retest_anchor_ids": sorted(
                row["anchor_id"] for row in local_anchors if row["anchor_id"] in global_retest
            ),
            "anchors": [],
            "reproducibility_checks": [],
        }
        atomic_json_dump(artifact, args.out)

    max_future = int(protocol["max_steps"])
    max_coordinates = (
        int(args.n_steps) * (1 + int(args.tokens_per_step))
        + int(args.max_action_tokens)
    )
    resets = len(local_anchors) * (1 + max_coordinates) * int(args.k) * 2
    resets += len(artifact["retest_anchor_ids"]) * (1 + max_coordinates) * int(args.k) * 2
    resets += 8
    env = make_repeated_env(manager, game_files[game_id], resets)
    try:
        for anchor_meta in sorted(local_anchors, key=lambda row: row["anchor_id"]):
            episode = episode_by_id[anchor_meta["episode_id"]]
            turn = episode["turns"][int(anchor_meta["turn_id"])]
            thought_trace = turn["decision_traces"]["thought"]
            action_trace = turn["decision_traces"]["action"]
            thought_selection = select_step_token_coordinates(
                thought_trace,
                anchor_meta["anchor_id"],
                args.selection_seed,
                block_size=args.block_size,
                n_steps=args.n_steps,
                tokens_per_step=args.tokens_per_step,
            )
            action_trie = build_action_trie(
                runtime.tokenizer, turn["state_before"]["admissible_commands"]
            )
            if not action_trie.is_complete(action_trace["response_ids"]):
                raise ValueError("base Action trace is not accepted by its frozen Trie")
            action_selection = select_action_token_coordinates(
                action_trace,
                anchor_meta["anchor_id"],
                args.selection_seed,
                action_trie.allowed_next,
                max_action_tokens=args.max_action_tokens,
                min_action_tokens=args.min_action_tokens,
            )
            selection = {
                "thought": thought_selection,
                "action": action_selection,
            }
            selection["selection_sha256"] = sha256_json(selection)
            coordinates = replacement_positions(thought_selection) + list(
                action_selection["coordinates"]
            )
            anchor_row = find_anchor_row(artifact, anchor_meta["anchor_id"])
            if anchor_row is None:
                anchor_row = {
                    **dict(anchor_meta),
                    "selection": selection,
                    "base_action": turn["executed_action"],
                    "repeats": {},
                }
                artifact["anchors"].append(anchor_row)
                atomic_json_dump(artifact, args.out)
            elif anchor_row.get("selection") != selection:
                raise ValueError("resumed anchor selection drift")
            repeat_ids = [0, 1] if anchor_meta["anchor_id"] in global_retest else [0]
            for repeat_id in repeat_ids:
                repeat = anchor_row["repeats"].setdefault(
                    str(repeat_id), {"originals": {}, "references": []}
                )
                schedules = {}
                for sample_id in range(int(args.k)):
                    schedule = make_crn_schedule(
                        args.branch_seed,
                        anchor_meta["anchor_id"],
                        repeat_id,
                        sample_id,
                        max(0, max_future - int(anchor_meta["turn_id"]) - 1),
                    )
                    schedules[sample_id] = schedule
                    key = str(sample_id)
                    if key not in repeat["originals"]:
                        repeat["originals"][key] = run_one_branch(
                            runtime,
                            env,
                            episode,
                            anchor_meta,
                            schedule,
                            "original",
                            {},
                            int(protocol["max_steps"]),
                            int(protocol["max_history"]),
                        )
                        atomic_json_dump(artifact, args.out)
                for coordinate in coordinates:
                    if find_reference(repeat, coordinate["coordinate_id"]) is not None:
                        continue
                    stored_pairs = []
                    summary_pairs = []
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
                                turn["state_before"]["admissible_commands"],
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
                                args.block_size,
                                args.gen_length,
                                args.logprob_tolerance,
                            )
                            action_seed = stable_seed(
                                schedule["intervention_seed"],
                                coordinate["coordinate_id"],
                                "action",
                            )
                            regenerated = action_from_modified_thought(
                                runtime,
                                thought_trace,
                                replacement["modified_response_ids"],
                                turn["state_before"]["admissible_commands"],
                                action_seed,
                            )
                        intervention = {
                            "kind": "explicit_counterfactual_action",
                            **regenerated,
                        }
                        alternative = run_one_branch(
                            runtime,
                            env,
                            episode,
                            anchor_meta,
                            schedule,
                            "alternative",
                            intervention,
                            int(protocol["max_steps"]),
                            int(protocol["max_history"]),
                        )
                        original = repeat["originals"][str(sample_id)]
                        effect = float(original["return"]) - float(alternative["return"])
                        pair = {
                            "sample_id": sample_id,
                            "crn_schedule": schedule,
                            "original_key": str(sample_id),
                            "replacement": replacement,
                            "regenerated_action": regenerated,
                            "action_changed": regenerated["action"] != turn["executed_action"],
                            "alternative": alternative,
                            "paired_effect": effect,
                        }
                        stored_pairs.append(pair)
                        summary_pairs.append(
                            {
                                "crn_schedule": schedule,
                                "original": original,
                                "alternative": alternative,
                                "paired_effect": effect,
                            }
                        )
                    reference = {
                        **coordinate,
                        "repeat_id": repeat_id,
                        "pairs": stored_pairs,
                        "reference": summarize_paired_effects(summary_pairs),
                    }
                    repeat["references"].append(reference)
                    atomic_json_dump(artifact, args.out)
                    print(
                        "[O02] game={} anchor={} repeat={} level={} step={} mean={:.4f}".format(
                            int(args.game_index),
                            anchor_meta["anchor_id"],
                            repeat_id,
                            coordinate["level"],
                            coordinate["step_id"],
                            reference["reference"]["mean_effect"],
                        ),
                        flush=True,
                    )
    finally:
        close_env(env)

    for anchor in artifact["anchors"]:
        expected_coordinates = (
            int(args.n_steps) * (1 + int(args.tokens_per_step))
            + len(anchor["selection"]["action"]["coordinates"])
        )
        expected_repeats = 2 if anchor["anchor_id"] in global_retest else 1
        if len(anchor["repeats"]) != expected_repeats:
            raise ValueError("anchor repeat count is incomplete")
        for repeat in anchor["repeats"].values():
            if len(repeat["originals"]) != int(args.k):
                raise ValueError("shared original count is incomplete")
            if len(repeat["references"]) != expected_coordinates:
                raise ValueError("coordinate reference count is incomplete")
            if any(len(row["pairs"]) != int(args.k) for row in repeat["references"]):
                raise ValueError("coordinate completion count is incomplete")
    if len(artifact["anchors"]) != len(local_anchors):
        raise ValueError("local anchor count is incomplete")
    artifact["complete"] = True
    artifact["n_anchors"] = len(artifact["anchors"])
    artifact["n_retest_anchors"] = len(artifact["retest_anchor_ids"])
    artifact["n_references"] = sum(
        len(repeat["references"])
        for anchor in artifact["anchors"]
        for repeat in anchor["repeats"].values()
    )
    artifact["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json_dump(artifact, args.out)
    print(
        "[O02] PASS game={} anchors={} refs={} out={}".format(
            args.game_index, artifact["n_anchors"], artifact["n_references"], args.out
        ),
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--model_dir", required=True)
    result.add_argument("--alfworld_config", required=True)
    result.add_argument("--prompt_json", required=True)
    result.add_argument("--split_manifest", required=True)
    result.add_argument("--counterfactual_seed_manifest", required=True)
    result.add_argument("--seed_phase", choices=("screening", "submission"), default="screening")
    result.add_argument("--seed_replicate", type=int, default=0)
    result.add_argument("--split_name", choices=("dev_seen", "final_unseen"), default="dev_seen")
    result.add_argument("--split_root")
    result.add_argument("--base_artifact", required=True)
    result.add_argument("--game_index", type=int, required=True)
    result.add_argument("--out", required=True)
    result.add_argument("--thought_order", choices=("ao", "ar"), default="ao")
    result.add_argument("--action_order", choices=("ao", "ar"), default="ar")
    result.add_argument("--action_grammar", choices=("none", "trie"), default="trie")
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--gen_length", type=int, default=128)
    result.add_argument("--action_gen_length", type=int, default=24)
    result.add_argument("--block_length", type=int, default=4)
    result.add_argument("--denoising_steps", type=int, default=4)
    result.add_argument("--max_history", type=int, default=24)
    result.add_argument("--max_steps", type=int, default=30)
    result.add_argument("--num_games", type=int, default=16)
    result.add_argument("--rollouts_per_game", type=int, default=2)
    result.add_argument("--base_rollout_seed", type=int, default=62001)
    result.add_argument("--branch_seed", type=int, default=72001)
    result.add_argument("--selection_seed", type=int, default=82001)
    result.add_argument("--k", type=int, default=4)
    result.add_argument("--n_steps", type=int, default=2)
    result.add_argument("--tokens_per_step", type=int, default=2)
    result.add_argument("--max_action_tokens", type=int, default=2)
    result.add_argument("--min_action_tokens", type=int, default=1)
    result.add_argument("--retest_fraction", type=float, default=0.25)
    result.add_argument("--logprob_tolerance", type=float, default=0.05)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.thought_order != "ao" or args.action_order != "ar" or args.action_grammar != "trie":
        raise ValueError("O02 requires AO Thought and canonical AR+Trie Action")
    if args.block_length != 4 or args.denoising_steps != 4:
        raise ValueError("O02 must match the frozen O01 Thought decoder")
    if (
        args.k < 1
        or args.n_steps < 1
        or args.tokens_per_step < 1
        or args.max_action_tokens < 1
        or not (1 <= args.min_action_tokens <= args.max_action_tokens)
    ):
        raise ValueError("O02 counts must be positive")
    args.seed_registration = validate_registered_seeds(args)
    runtime = Runtime(args)
    run(args, runtime)


if __name__ == "__main__":
    main()
