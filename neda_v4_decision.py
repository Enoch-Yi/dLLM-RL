#!/usr/bin/env python3
"""V4 two-stage decoder with replayable constrained AR Action support.

This module deliberately leaves the frozen V3/O01 decoder untouched.  Its only
semantic extension is recording the exact trie support used at every sampled
Action token so a learner can normalize old and new scores over the identical
post-constraint distribution.
"""

import math

import torch
import torch.nn.functional as F

from neda_data_contract import (
    EXACT_ACTION_SCORING_LAYOUT,
    NATIVE_THOUGHT_REPLAY_CONTRACT_VERSION,
    NATIVE_THOUGHT_SCORING_LAYOUT,
    STRUCTURAL_ACTION_DELIMITER,
    build_structural_action_prefix,
    make_decision_trace,
    make_sampling_spec,
    token_boundary_at_char,
    trim_generation_trace,
)
from neda_joint_policy import (
    JOINT_METHODS,
    POSITION_TRACE_CONTRACT_VERSION,
    mapg_position_scores,
    method_position_policy,
    sample_plackett_luce,
)
from neda_repro import sha256_json
from neda_torch_replay import exact_replay_numerics, make_basic_block_attention
from r002_alfworld import (
    _as_list,
    _num_transfer,
    _sample,
    _stop_sequences,
    build_action_trie,
    two_stage_decision_decode as legacy_two_stage_decision_decode,
)
from neda_torch_replay import make_absolute_block_duplicate_attention


V4_DECISION_CONTRACT_VERSION = "neda-v4-decision-trace-v1"


def _planned_rounds(
    prompt_length,
    response_width,
    block_length,
    denoising_steps,
):
    first_block = int(prompt_length) // int(block_length)
    final_block = (
        int(prompt_length) + int(response_width)
    ) // int(block_length)
    return max(1, (final_block - first_block) * int(denoising_steps))


def _even_boundaries(round_ids, count):
    values = sorted(set(int(value) for value in round_ids if int(value) >= 0))
    if not values:
        return []
    count = min(max(1, int(count)), len(values))
    if count == 1:
        return [values[-1]]
    indices = {
        round(index * (len(values) - 1) / (count - 1))
        for index in range(count)
    }
    return [values[index] for index in sorted(indices)]


@torch.no_grad()
def _score_action_under_partial_thought(
    model,
    thought_trace,
    action_trace,
    mask_id,
    boundary_count,
):
    """Score the executed Action under StepMerge Thought prefixes.

    This is not presented as ground-truth causal attribution.  It measures how
    much each segment of actual denoising commitments changes the behavior
    policy's teacher-forced log-probability for the Action that was eventually
    submitted to ALFWorld.  Segment deltas are converted to non-negative,
    normalized allocation weights by the credit materializer.
    """

    response_ids = [int(value) for value in thought_trace["response_ids"]]
    step_map = [int(value) for value in thought_trace["step_map"]]
    thought_prefix = [int(value) for value in thought_trace["prefix_ids"]]
    action_prefix = [int(value) for value in action_trace["prefix_ids"]]
    if action_prefix[: len(thought_prefix)] != thought_prefix:
        raise ValueError("Action/Thought prefixes do not share the environment state")
    boundaries = _even_boundaries(step_map, boundary_count)
    if not boundaries:
        if response_ids or step_map:
            raise ValueError(
                "NeDA Thought tokens and committed rounds are misaligned"
            )
        return {
            "contract_version": "neda-action-evidence-v1",
            "estimator": "not applicable: Action-only decision",
            "interpretation": (
                "No synthetic Thought credit; the environment-turn signal is "
                "assigned only to the recorded Action policy coordinate"
            ),
            "allocation_status": "ACTION_ONLY",
            "boundaries": [],
            "sequence_logprobs": [],
            "token_logprobs": [],
            "segments": [],
        }
    thought_begin = len(thought_prefix)
    thought_end = thought_begin + len(response_ids)
    if action_prefix[thought_begin:thought_end] != response_ids:
        raise ValueError("Action prefix does not contain the recorded Thought")

    action_ids = [int(value) for value in action_trace["response_ids"]]
    width = int(action_trace["replay_width"])
    allowed_rows = action_trace["constraint_allowed_token_ids"]
    prefix_length = len(action_prefix)
    attention = make_basic_block_attention(
        prefix_length + 2 * width,
        prefix_length,
        block_size=1,
        device=model.device,
    )
    original = torch.arange(
        prefix_length + width, dtype=torch.long, device=model.device
    ).unsqueeze(0)
    position_ids = torch.cat(
        [original, original[:, prefix_length:]], dim=1
    )
    score_boundaries = [-1] + boundaries
    sequence_scores = []
    token_scores = []
    for boundary in score_boundaries:
        partial = [
            token if round_id <= boundary else int(mask_id)
            for token, round_id in zip(response_ids, step_map)
        ]
        prefix = list(action_prefix)
        prefix[thought_begin:thought_end] = partial
        response = action_ids + [int(mask_id)] * (width - len(action_ids))
        sequence = torch.as_tensor(
            [prefix + response + response],
            dtype=torch.long,
            device=model.device,
        )
        logits = model(
            input_ids=sequence,
            attention_mask=attention,
            position_ids=position_ids,
            use_cache=False,
            store_kv=False,
        ).logits
        projected = torch.cat(
            [
                logits[:, :prefix_length, :],
                logits[:, prefix_length + width :, :],
            ],
            dim=1,
        )
        local = []
        for offset, (target, allowed) in enumerate(
            zip(action_ids, allowed_rows)
        ):
            values = [int(value) for value in allowed]
            if int(target) not in values:
                raise ValueError("recorded Action token left its trie support")
            selected_logits = projected[
                0, prefix_length + offset, :
            ].float().index_select(
                0,
                torch.as_tensor(values, dtype=torch.long, device=model.device),
            )
            local.append(
                F.log_softmax(selected_logits, dim=0)[values.index(int(target))]
            )
        per_token = torch.stack(local)
        token_scores.append([float(value) for value in per_token.cpu()])
        sequence_scores.append(float(per_token.mean().cpu()))

    segments = []
    previous_boundary = -1
    for index, boundary in enumerate(boundaries):
        member_rounds = [
            value
            for value in sorted(set(step_map))
            if previous_boundary < int(value) <= int(boundary)
        ]
        if not member_rounds:
            raise ValueError("NeDA StepMerge Action-evidence segment is empty")
        segments.append(
            {
                "segment_id": index,
                "left_round_exclusive": int(previous_boundary),
                "right_round_inclusive": int(boundary),
                "member_rounds": member_rounds,
                "action_logprob_before": sequence_scores[index],
                "action_logprob_after": sequence_scores[index + 1],
                "action_logprob_delta": (
                    sequence_scores[index + 1] - sequence_scores[index]
                ),
            }
        )
        previous_boundary = boundary
    # If StepMerge sampled fewer boundaries than realized rounds, the final
    # boundary is always included by _even_boundaries, hence all rounds occur.
    covered = [value for segment in segments for value in segment["member_rounds"]]
    if covered != sorted(set(step_map)):
        raise ValueError("NeDA Action-evidence segments do not cover all rounds")
    return {
        "contract_version": "neda-action-evidence-v1",
        "estimator": "teacher-forced executed-Action log-probability delta",
        "interpretation": "StepMerge allocation signal, not ground-truth causal credit",
        "boundaries": boundaries,
        "sequence_logprobs": sequence_scores,
        "token_logprobs": token_scores,
        "segments": segments,
    }


@torch.no_grad()
def exact_fixed_width_ao_thought_generate(
    model,
    input_ids,
    mask_id,
    gen_length=128,
    block_length=4,
    denoising_steps=4,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    confidence_threshold=0.85,
    stop_ids=None,
    stop_sequences=None,
    position_policy="confidence",
    position_temperature=0.5,
    position_head=None,
):
    """Sample AO Thought in exactly the fixed learner scoring layout.

    The historical fast generator sampled a four-token query against a KV
    cache, while the differentiable learner replayed a full duplicated tensor.
    Those paths are mathematically close but produced material log-probability
    drift on uncertain tokens.  V4 therefore defines the behavior policy in
    the same fixed-width, absolute-block duplicated layout used by the learner.
    Future response positions remain MASK and are hidden by block attention.
    """

    model.eval()
    prompt_length = int(input_ids.shape[1])
    block_length = int(block_length)
    denoising_steps = int(denoising_steps)
    if block_length <= 0 or denoising_steps <= 0 or int(gen_length) <= 0:
        raise ValueError("invalid fixed-width AO Thought generation dimensions")
    total = (
        (prompt_length + int(gen_length) + block_length - 1) // block_length
    ) * block_length
    response_width = total - prompt_length
    sequence = torch.full(
        (1, total), int(mask_id), dtype=torch.long, device=model.device
    )
    sequence[:, :prompt_length] = input_ids
    commit_round = torch.full(
        (response_width,), -1, dtype=torch.long, device=model.device
    )
    commit_confidence = torch.zeros(
        (response_width,), dtype=torch.float32, device=model.device
    )
    behavior_logprobs = torch.full(
        (response_width,), -torch.inf, dtype=torch.float32, device=model.device
    )
    position_trace = []
    attention = make_absolute_block_duplicate_attention(
        prompt_length + 2 * response_width,
        prompt_length,
        block_length,
        device=model.device,
    )
    original_positions = torch.arange(
        total, dtype=torch.long, device=model.device
    ).unsqueeze(0)
    position_ids = torch.cat(
        [original_positions, original_positions[:, prompt_length:]], dim=1
    )
    sampling = make_sampling_spec(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        constraint="none",
    )
    transfer = _num_transfer(block_length, denoising_steps).to(model.device)
    first_block = prompt_length // block_length
    num_blocks = total // block_length
    total_planned_rounds = _planned_rounds(
        prompt_length,
        response_width,
        block_length,
        denoising_steps,
    )
    round_id = 0

    for block_index in range(first_block, num_blocks):
        block_start = block_index * block_length
        response_start = max(prompt_length, block_start)
        block_end = (block_index + 1) * block_length
        response_slice = slice(response_start, block_end)
        for step in range(denoising_steps + 1):
            current = sequence[:, response_slice]
            masked = current.eq(int(mask_id))
            if not bool(masked.any()):
                break
            extended = torch.cat([sequence, sequence[:, prompt_length:]], dim=1)
            query_start = total + response_start - prompt_length
            query_end = total + block_end - prompt_length
            with exact_replay_numerics():
                if position_policy == "dcolt_upm":
                    if position_head is None:
                        raise ValueError("DCoLT generation requires a loaded UPM")
                    outputs = model.model(
                        input_ids=extended,
                        attention_mask=attention,
                        position_ids=position_ids,
                        use_cache=False,
                        store_kv=False,
                    )
                    full_hidden = outputs.last_hidden_state
                    full_logits = model.lm_head(full_hidden)
                    response_hidden = full_hidden[:, total:, :]
                    mask_index = sequence[:, prompt_length:].eq(int(mask_id))
                    current_block = torch.zeros_like(mask_index)
                    current_block[
                        :,
                        response_start - prompt_length :
                        block_end - prompt_length,
                    ] = True
                    timestep = torch.full(
                        (sequence.shape[0],),
                        float(round_id) / float(total_planned_rounds),
                        dtype=torch.float32,
                        device=model.device,
                    )
                    full_position_scores = position_head(
                        response_hidden,
                        timestep,
                        mask_index,
                        current_block,
                    )
                    logits = full_logits[:, query_start:query_end]
                else:
                    full_logits = model(
                        input_ids=extended,
                        attention_mask=attention,
                        position_ids=position_ids,
                        use_cache=False,
                        store_kv=False,
                    ).logits
                    logits = full_logits[:, query_start:query_end]
                    full_position_scores = mapg_position_scores(
                        full_logits[:, total:, :]
                    )
            sampled, probability, logprob = _sample(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            confidence = torch.where(masked, probability, -torch.inf)
            selected = torch.zeros_like(sampled, dtype=torch.bool)
            required = min(
                int(transfer[step].item()), int(masked[0].sum().item())
            )
            if required <= 0:
                round_id += 1
                continue
            if position_policy in ("mapg_logit", "dcolt_upm"):
                if sequence.shape[0] != 1:
                    raise ValueError("recorded position policy currently requires batch one")
                candidates_local = masked[0].nonzero(as_tuple=True)[0].tolist()
                candidates = [
                    response_start - prompt_length + int(value)
                    for value in candidates_local
                ]
                chosen, position_logprob = sample_plackett_luce(
                    full_position_scores[0],
                    candidates,
                    required,
                    float(position_temperature),
                )
                chosen_local = [
                    int(value) - (response_start - prompt_length)
                    for value in chosen
                ]
                selected[
                    0,
                    torch.as_tensor(
                        chosen_local, dtype=torch.long, device=model.device
                    ),
                ] = True
                position_trace.append(
                    {
                        "contract_version": POSITION_TRACE_CONTRACT_VERSION,
                        "round_id": int(round_id),
                        "policy": str(position_policy),
                        "temperature": float(position_temperature),
                        "candidate_positions": candidates,
                        "selected_positions": chosen,
                        "behavior_logprob": float(position_logprob),
                        "current_block_positions": list(
                            range(
                                response_start - prompt_length,
                                block_end - prompt_length,
                            )
                        ),
                        "timestep": (
                            float(round_id) / float(total_planned_rounds)
                        ),
                    }
                )
            else:
                if position_policy != "confidence":
                    raise ValueError("unknown Thought position policy")
                for row_index in range(confidence.shape[0]):
                    high = confidence[row_index] > float(confidence_threshold)
                    if int(high.sum().item()) >= required:
                        selected[row_index] = high
                    else:
                        _, indices = torch.topk(confidence[row_index], required)
                        selected[row_index, indices] = True
            newly = selected[0].nonzero(as_tuple=True)[0]
            if newly.numel() > 0:
                offsets = response_start - prompt_length + newly
                commit_round[offsets] = int(round_id)
                commit_confidence[offsets] = confidence[0, newly].detach().float()
                behavior_logprobs[offsets] = logprob[0, newly].detach().float()
            current[selected] = sampled[selected]
            sequence[:, response_slice] = current
            round_id += 1

        generated = sequence[0, prompt_length:].tolist()
        hit_stop_id = stop_ids is not None and any(
            int(value) != int(mask_id)
            and int(value) in {int(stop) for stop in stop_ids if stop is not None}
            for value in generated
        )
        hit_stop_sequence = any(
            list(candidate)
            and any(
                generated[index : index + len(candidate)]
                == [int(value) for value in candidate]
                for index in range(max(0, len(generated) - len(candidate) + 1))
            )
            for candidate in (stop_sequences or [])
        )
        if hit_stop_id or hit_stop_sequence:
            break

    return sequence[0], {
        "step_map": commit_round.cpu(),
        "commit_confidence": commit_confidence.cpu(),
        "behavior_logprobs": behavior_logprobs.cpu(),
        "sampling": sampling,
        "replay_width": int(response_width),
        "scoring_layout": NATIVE_THOUGHT_SCORING_LAYOUT,
        "position_policy": str(position_policy),
        "position_trace": position_trace,
    }


@torch.no_grad()
def exact_constrained_ar_action_generate(
    model,
    input_ids,
    mask_id,
    constraint,
    constraint_name="trie",
    gen_length=24,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
):
    if constraint is None or constraint_name != "trie":
        raise ValueError("V4 constrained Action generation requires a trie")
    model.eval()
    response_width = int(gen_length)
    if response_width <= 0:
        raise ValueError("exact AR Action generation requires gen_length > 0")
    sampling = make_sampling_spec(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        constraint=constraint_name,
    )
    prefix_length = int(input_ids.shape[1])
    sequence = torch.full(
        (1, prefix_length + response_width),
        int(mask_id),
        dtype=torch.long,
        device=model.device,
    )
    sequence[:, :prefix_length] = input_ids
    commit_round = torch.full(
        (response_width,), -1, dtype=torch.long, device=model.device
    )
    commit_confidence = torch.zeros(
        (response_width,), dtype=torch.float32, device=model.device
    )
    behavior_logprobs = torch.full(
        (response_width,), -torch.inf, dtype=torch.float32, device=model.device
    )
    allowed_token_ids = [None] * response_width
    attention = make_basic_block_attention(
        prefix_length + 2 * response_width,
        prefix_length,
        block_size=1,
        device=model.device,
    )
    base_positions = torch.arange(
        prefix_length + response_width, dtype=torch.long, device=model.device
    ).unsqueeze(0)
    position_ids = torch.cat(
        [base_positions, base_positions[:, prefix_length:]], dim=1
    )
    for offset in range(response_width):
        committed = sequence[0, prefix_length : prefix_length + offset].tolist()
        allowed = sorted(int(value) for value in constraint(committed))
        if not allowed:
            break
        allowed_token_ids[offset] = allowed
        extended = torch.cat([sequence, sequence[:, prefix_length:]], dim=1)
        with exact_replay_numerics():
            logits = model(
                input_ids=extended,
                attention_mask=attention,
                position_ids=position_ids,
                use_cache=False,
                store_kv=False,
            ).logits
        query = logits[
            :, prefix_length + response_width + offset :
            prefix_length + response_width + offset + 1, :
        ]
        vocabulary_mask = torch.full(
            (query.shape[-1],), float("-inf"), device=query.device
        )
        vocabulary_mask[
            torch.as_tensor(allowed, dtype=torch.long, device=query.device)
        ] = 0.0
        token, confidence, logp = _sample(
            query + vocabulary_mask,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        token_id = int(token[0, 0])
        sequence[0, prefix_length + offset] = token_id
        commit_round[offset] = offset
        commit_confidence[offset] = confidence[0, 0].detach().float()
        behavior_logprobs[offset] = logp[0, 0].detach().float()
    return sequence[0], {
        "step_map": commit_round.cpu(),
        "commit_confidence": commit_confidence.cpu(),
        "behavior_logprobs": behavior_logprobs.cpu(),
        "sampling": sampling,
        "replay_width": response_width,
        "scoring_layout": EXACT_ACTION_SCORING_LAYOUT,
        "constraint_allowed_token_ids": allowed_token_ids,
    }


def _slice(trace, end):
    end = int(end)
    result = {
        key: list(trace[key][:end])
        for key in (
            "response_ids", "step_map", "behavior_logprobs", "commit_confidence"
        )
    }
    result["sampling"] = dict(trace["sampling"])
    result["replay_width"] = int(trace["replay_width"])
    result["scoring_layout"] = str(trace["scoring_layout"])
    result["constraint_allowed_token_ids"] = [
        list(values) for values in trace["constraint_allowed_token_ids"][:end]
    ]
    return result


def _attach_constraint_contract(action_trace, allowed_rows):
    result = make_decision_trace(
        action_trace["prefix_ids"],
        action_trace,
        action_trace["text"],
        action_trace["tokenizer"],
        kind="action",
    )
    result["constraint_allowed_token_ids"] = [list(values) for values in allowed_rows]
    result["decision_contract_version"] = V4_DECISION_CONTRACT_VERSION
    result["trace_sha256"] = sha256_json(
        {
            "prefix_ids": result["prefix_ids"],
            "response_ids": result["response_ids"],
            "step_map": result["step_map"],
            "behavior_logprobs": result["behavior_logprobs"],
            "sampling": result["sampling"],
            "replay_width": result["replay_width"],
            "scoring_layout": result["scoring_layout"],
            "constraint_allowed_token_ids": result["constraint_allowed_token_ids"],
        }
    )
    return result


def _attach_native_thought_replay(
    decision_trace,
    raw_output,
    raw_trace,
    prompt_length,
    mask_id,
    block_size,
):
    """Keep the complete sampled block while optimizing interface Thought only."""

    response_ids = _as_list(raw_output[int(prompt_length) :])
    step_map = _as_list(raw_trace["step_map"])
    behavior = [float(value) for value in _as_list(raw_trace["behavior_logprobs"])]
    confidence = [float(value) for value in _as_list(raw_trace["commit_confidence"])]
    lengths = {len(response_ids), len(step_map), len(behavior), len(confidence)}
    if len(lengths) != 1:
        raise ValueError("native Thought raw trace arrays are not aligned")
    if (
        int(raw_trace.get("replay_width", 0)) != len(response_ids)
        or raw_trace.get("scoring_layout") != NATIVE_THOUGHT_SCORING_LAYOUT
    ):
        raise ValueError("native Thought behavior scorer identity drift")
    replay_end = 0
    for token, step, logp in zip(response_ids, step_map, behavior):
        if int(token) == int(mask_id) or int(step) < 0 or not math.isfinite(logp):
            break
        replay_end += 1
    policy_width = len(decision_trace["response_ids"])
    if replay_end < policy_width or response_ids[:policy_width] != list(
        decision_trace["response_ids"]
    ):
        raise ValueError("native Thought replay does not cover the interface Thought")
    if (int(prompt_length) + replay_end) % int(block_size) != 0:
        raise ValueError("native Thought replay ended inside an absolute diffusion block")
    native = {
        "contract_version": NATIVE_THOUGHT_REPLAY_CONTRACT_VERSION,
        "scoring_layout": NATIVE_THOUGHT_SCORING_LAYOUT,
        "block_size": int(block_size),
        "replay_width": int(raw_trace["replay_width"]),
        "response_ids": [int(value) for value in response_ids[:replay_end]],
        "step_map": [int(value) for value in step_map[:replay_end]],
        "behavior_logprobs": behavior[:replay_end],
        "commit_confidence": confidence[:replay_end],
        "optimization_mask": [
            index < policy_width for index in range(replay_end)
        ],
    }
    position_policy = str(raw_trace.get("position_policy", "confidence"))
    native["position_policy"] = position_policy
    if position_policy != "confidence":
        kept_rounds = {
            int(value) for value in step_map[:replay_end] if int(value) >= 0
        }
        position_trace = [
            dict(value)
            for value in raw_trace.get("position_trace", [])
            if int(value["round_id"]) in kept_rounds
        ]
        if {int(value["round_id"]) for value in position_trace} != kept_rounds:
            raise ValueError("native Thought position trace does not cover replay rounds")
        for value in position_trace:
            candidates = [int(item) for item in value["candidate_positions"]]
            selected = [int(item) for item in value["selected_positions"]]
            if any(item >= replay_end for item in selected):
                raise ValueError("position trace selected a token outside native replay")
            value["candidate_positions"] = candidates
            value["selected_positions"] = selected
        native["position_trace"] = position_trace
    native["replay_sha256"] = sha256_json(native)
    result = dict(decision_trace)
    result["native_replay"] = native
    result["decision_contract_version"] = V4_DECISION_CONTRACT_VERSION
    trace_body = dict(result)
    trace_body.pop("trace_sha256", None)
    result["trace_sha256"] = sha256_json(trace_body)
    return result


def two_stage_decision_decode(
    model,
    tokenizer,
    prompt_ids,
    admissible,
    mask_id,
    stop_ids,
    thought_order="ao",
    action_order="ar",
    action_grammar="none",
    gen_length=128,
    action_gen_length=24,
    block_length=4,
    denoising_steps=4,
    temperature=0.3,
    rl_method=None,
    position_temperature=0.5,
    position_head=None,
    neda_credit_boundaries=4,
    collect_neda_action_evidence=True,
):
    if action_grammar != "trie":
        return legacy_two_stage_decision_decode(
            model, tokenizer, prompt_ids, admissible, mask_id, stop_ids,
            thought_order=thought_order, action_order=action_order,
            action_grammar=action_grammar, gen_length=gen_length,
            action_gen_length=action_gen_length, block_length=block_length,
            denoising_steps=denoising_steps, temperature=temperature,
        )
    if thought_order not in ("ao", "ar") or action_order != "ar":
        raise ValueError("V4 trie interface requires AO/AR Thought and AR Action")
    if rl_method is not None:
        rl_method = str(rl_method).lower()
        if rl_method not in JOINT_METHODS:
            raise ValueError("unknown joint RL method")
        if thought_order != "ao":
            raise ValueError("MAPG/DCoLT/NeDA require AO Thought")
        position_policy = method_position_policy(rl_method)
    else:
        position_policy = "confidence"
    prompt_ids = [int(token) for token in prompt_ids]
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    thought_block_size = block_length if thought_order == "ao" else 1
    thought_out, thought_raw_trace = exact_fixed_width_ao_thought_generate(
        model,
        prompt_tensor,
        mask_id,
        gen_length=gen_length,
        block_length=thought_block_size,
        denoising_steps=denoising_steps if thought_order == "ao" else 1,
        temperature=temperature,
        confidence_threshold=0.85 if thought_order == "ao" else 0.0,
        stop_ids=stop_ids,
        stop_sequences=_stop_sequences(tokenizer, ("\nAction:", "Action:")),
        position_policy=position_policy,
        position_temperature=position_temperature,
        position_head=position_head,
    )
    thought_full = trim_generation_trace(
        _as_list(thought_out[len(prompt_ids) :]),
        _as_list(thought_raw_trace["step_map"]),
        _as_list(thought_raw_trace["behavior_logprobs"]),
        mask_id,
        stop_ids,
        confidence=_as_list(thought_raw_trace["commit_confidence"]),
        sampling=thought_raw_trace["sampling"],
    )
    thought_full_text = tokenizer.decode(
        thought_full["response_ids"], skip_special_tokens=True
    )
    structural = build_structural_action_prefix(
        tokenizer, prompt_ids, thought_full_text, thought_full["response_ids"]
    )
    thought_end = int(structural["thought_end"])
    thought_trace = {
        key: list(thought_full[key][:thought_end])
        for key in (
            "response_ids", "step_map", "behavior_logprobs", "commit_confidence"
        )
    }
    thought_trace["sampling"] = dict(thought_full["sampling"])
    thought_text = tokenizer.decode(
        thought_trace["response_ids"], skip_special_tokens=True
    )

    trie = build_action_trie(tokenizer, admissible)
    action_prefix_ids = list(structural["action_prefix_ids"])
    action_tensor = torch.tensor(
        [action_prefix_ids], dtype=torch.long, device=model.device
    )
    action_out, action_raw = exact_constrained_ar_action_generate(
        model,
        action_tensor,
        mask_id,
        trie.allowed_next,
        gen_length=action_gen_length,
        temperature=temperature,
    )
    action_trace = trim_generation_trace(
        _as_list(action_out[len(action_prefix_ids) :]),
        _as_list(action_raw["step_map"]),
        _as_list(action_raw["behavior_logprobs"]),
        mask_id,
        stop_ids,
        confidence=_as_list(action_raw["commit_confidence"]),
        sampling=action_raw["sampling"],
    )
    action_trace["replay_width"] = int(action_raw["replay_width"])
    action_trace["scoring_layout"] = str(action_raw["scoring_layout"])
    action_trace["constraint_allowed_token_ids"] = [
        list(values)
        for values in action_raw["constraint_allowed_token_ids"][
            : len(action_trace["response_ids"])
        ]
    ]
    action_text = tokenizer.decode(
        action_trace["response_ids"], skip_special_tokens=True
    )
    if "\n" in action_text:
        end = token_boundary_at_char(
            tokenizer,
            action_text,
            action_trace["response_ids"],
            action_text.index("\n") + 1,
        )
        action_trace = _slice(action_trace, end)
        action_text = tokenizer.decode(
            action_trace["response_ids"], skip_special_tokens=True
        )
    raw_action = action_text.split("\n", 1)[0].strip()
    if raw_action.endswith("."):
        raw_action = raw_action[:-1].strip()
    if not raw_action or raw_action not in admissible:
        raise ValueError("trie decoder did not terminate on an admissible Action")

    response_ids = (
        thought_trace["response_ids"]
        + list(structural["delimiter_ids"])
        + action_trace["response_ids"]
    )
    action_decision = _attach_constraint_contract(
        {
            **action_trace,
            "prefix_ids": action_prefix_ids,
            "text": action_text,
            "tokenizer": tokenizer,
        },
        action_trace["constraint_allowed_token_ids"],
    )
    thought_decision = _attach_native_thought_replay(
        make_decision_trace(
            prompt_ids, thought_trace, thought_text, tokenizer, kind="thought"
        ),
        thought_out,
        thought_raw_trace,
        len(prompt_ids),
        mask_id,
        thought_block_size,
    )
    if rl_method == "neda" and bool(collect_neda_action_evidence):
        thought_decision["action_evidence"] = _score_action_under_partial_thought(
            model,
            thought_decision,
            action_decision,
            mask_id,
            neda_credit_boundaries,
        )
        trace_body = dict(thought_decision)
        trace_body.pop("trace_sha256", None)
        thought_decision["trace_sha256"] = sha256_json(trace_body)
    return {
        "response_text": tokenizer.decode(response_ids, skip_special_tokens=True),
        "response_ids": response_ids,
        "raw_action": raw_action,
        "decision_traces": {
            "thought": thought_decision,
            "action": action_decision,
        },
        "thought_order": thought_order,
        "action_order": action_order,
        "action_grammar": action_grammar,
        "rl_method": rl_method,
        "position_policy": position_policy,
        "position_temperature": (
            float(position_temperature) if rl_method is not None else None
        ),
        "marker_injected": not structural["marker_generated"],
        "structural_action_delimiter": STRUCTURAL_ACTION_DELIMITER,
        "structural_action_delimiter_ids": structural["delimiter_ids"],
        "structural_marker_rebuilt": True,
        "action_only_decision": not bool(thought_trace["response_ids"]),
        "discarded_thought_pass_tokens": len(thought_full["response_ids"])
        - len(thought_trace["response_ids"]),
        "decision_boundary": {
            "contract_version": structural["boundary_contract"],
            "action_only_decision": not bool(thought_trace["response_ids"]),
            "marker_generated": bool(structural["marker_generated"]),
            "thought_full_token_count": len(
                thought_decision["native_replay"]["response_ids"]
            ),
            "thought_trimmed_generation_token_count": len(
                thought_full["response_ids"]
            ),
            "thought_native_replay_token_count": len(
                thought_decision["native_replay"]["response_ids"]
            ),
            "thought_end": int(structural["thought_end"]),
            "removed_trailing_whitespace_tokens": int(
                structural["removed_trailing_whitespace_tokens"]
            ),
            "alignment_source": structural["alignment_source"],
            "dropped_boundary_overlap_text": structural[
                "dropped_boundary_overlap_text"
            ],
            "dropped_boundary_overlap_kind": structural[
                "dropped_boundary_overlap_kind"
            ],
            "dropped_boundary_overlap_token_index": structural[
                "dropped_boundary_overlap_token_index"
            ],
            "dropped_boundary_overlap_token_id": structural[
                "dropped_boundary_overlap_token_id"
            ],
        },
        "decision_contract_version": V4_DECISION_CONTRACT_VERSION,
    }
