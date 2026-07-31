"""Exact-ID batching, deterministic masks, and exact commit-path replay."""

import copy
import random

from neda_data_contract import validate_decision_trace
from neda_repro import stable_seed


REPLAY_CONTRACT_VERSION = "neda-exact-replay-v1"


def trace_from_record(record, source="response", require_exact=True):
    traces = record.get("decision_traces", {})
    if source in traces:
        trace = copy.deepcopy(traces[source])
    elif source == "response" and all(
        key in record for key in ("prompt_ids", "response_ids", "step_map")
    ):
        trace = {
            "kind": "response",
            "prefix_ids": list(record["prompt_ids"]),
            "response_ids": list(record["response_ids"]),
            "step_map": list(record["step_map"]),
            "behavior_logprobs": list(
                record.get("behavior_logprobs", [0.0] * len(record["response_ids"]))
            ),
            "thought_span": list(record.get("thought_span", [0, len(record["response_ids"])])),
            "action_span": list(record.get("action_span", [0, 0])),
        }
    else:
        raise ValueError("record has no exact '{}' trace".format(source))
    validate_decision_trace(
        trace,
        require_logprobs=require_exact,
        require_sampling=require_exact,
        exact_replay=require_exact,
    )
    return trace


def deterministic_record_order(records, sample_order_seed):
    if sample_order_seed is None:
        return list(records)
    return sorted(
        records,
        key=lambda row: (
            stable_seed(int(sample_order_seed), row.get("sample_id", "missing"), "order"),
            str(row.get("sample_id", "")),
        ),
    )


def build_exact_lm_batch(
    records,
    pad_id,
    max_prompt_len,
    max_response_len,
    trace_source="response",
    require_exact=True,
    mask_id=None,
):
    """Pad stored IDs without decode/re-tokenize; return aligned trace arrays."""

    kept = []
    dropped = []
    for index, record in enumerate(records):
        trace = trace_from_record(record, trace_source, require_exact=require_exact)
        p_len = len(trace["prefix_ids"])
        r_len = len(trace["response_ids"])
        replay_width = int(trace.get("replay_width", r_len))
        if (
            p_len > int(max_prompt_len)
            or replay_width > int(max_response_len)
            or r_len == 0
        ):
            dropped.append(index)
            continue
        kept.append((index, record, trace))
    if not kept:
        raise ValueError("no exact trace survives length filtering")
    prompt_width = max(len(trace["prefix_ids"]) for _, _, trace in kept)
    response_width = max(
        int(trace.get("replay_width", len(trace["response_ids"])))
        for _, _, trace in kept
    )
    rows = []
    for original_index, record, trace in kept:
        prompt = list(trace["prefix_ids"])
        response = list(trace["response_ids"])
        left_pad = [int(pad_id)] * (prompt_width - len(prompt))
        right_pad_n = response_width - len(response)
        response_pad_id = int(mask_id) if (
            mask_id is not None and trace.get("scoring_layout") is not None
        ) else int(pad_id)
        input_ids = left_pad + prompt + response + [response_pad_id] * right_pad_n
        step_map = list(trace["step_map"]) + [-1] * right_pad_n
        behavior_logprobs = list(trace["behavior_logprobs"]) + [0.0] * right_pad_n
        action_start, action_end = trace.get("action_span", [0, 0])
        action_mask = [False] * response_width
        for position in range(int(action_start), int(action_end)):
            action_mask[position] = True
        rows.append(
            {
                "original_index": original_index,
                "sample_id": record.get("sample_id", str(original_index)),
                "input_ids": input_ids,
                "step_map": step_map,
                "behavior_logprobs": behavior_logprobs,
                "sampling": copy.deepcopy(trace.get("sampling")),
                "scoring_layout": trace.get("scoring_layout"),
                "replay_width": int(
                    trace.get("replay_width", len(trace["response_ids"]))
                ),
                "action_mask": action_mask,
                "response_length": len(response),
                "record": record,
            }
        )
    return {
        "contract_version": REPLAY_CONTRACT_VERSION,
        "start_pos": prompt_width,
        "response_width": response_width,
        "rows": rows,
        "dropped_indices": dropped,
    }


def deterministic_random_mask(length, lower, upper, mask_seed, sample_id):
    if length < 0:
        raise ValueError("length must be non-negative")
    if not (0.0 <= lower <= upper <= 1.0):
        raise ValueError("mask probability bounds must satisfy 0 <= lower <= upper <= 1")
    rng = random.Random(stable_seed(int(mask_seed), sample_id, "learner-mask"))
    return [rng.random() <= rng.uniform(lower, upper) for _ in range(length)]


def exact_replay_rows(input_ids, start_pos, response_width, step_map, mask_id, action_mask=None):
    """Reconstruct the state immediately before every global commit round.

    Unlike legacy TraceRL's per-block minimum, this groups tokens by the exact
    global commit round stored by behavior generation. Tokens with step >= r are
    masked when predicting the tokens committed at round r.
    """

    total = int(start_pos) + int(response_width)
    if len(input_ids) != total:
        raise ValueError("input_ids length does not match start_pos + response_width")
    if len(step_map) != response_width:
        raise ValueError("step_map length mismatch")
    if action_mask is None:
        action_mask = [True] * response_width
    if len(action_mask) != response_width:
        raise ValueError("action_mask length mismatch")
    rounds = sorted(set(int(step) for step in step_map if int(step) >= 0))
    rows = []
    response = list(input_ids[start_pos:])
    for round_id in rounds:
        selected = [
            int(step_map[position]) == round_id and bool(action_mask[position])
            for position in range(response_width)
        ]
        if not any(selected):
            continue
        learner_state = [
            int(mask_id) if int(step_map[position]) >= round_id else response[position]
            for position in range(response_width)
        ]
        # Both response copies must be the recorded state at round r.  Keeping
        # final/current/future truth tokens in the first copy is mathematically
        # hidden by the block-causal mask, but makes the actual behavior and
        # learner input tensors different and produced measurable kernel drift.
        model_state = list(input_ids[:start_pos]) + learner_state
        rows.append(
            {
                "round_id": round_id,
                "extended_input_ids": model_state + learner_state,
                "label_ids": list(input_ids),
                "prediction_mask": [False] * start_pos + selected,
            }
        )
    return rows
