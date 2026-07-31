#!/usr/bin/env python3
"""Torch-free construction of native AO-Thought/AR-Action replay rows."""

import copy
import json
import math
from typing import Any, Dict, List, Tuple

from neda_data_contract import (
    EXACT_ACTION_SCORING_LAYOUT,
    NATIVE_THOUGHT_REPLAY_CONTRACT_VERSION,
    NATIVE_THOUGHT_SCORING_LAYOUT,
    validate_decision_trace,
)
from neda_replay import exact_replay_rows
from neda_repro import stable_seed


MULTITRACE_CONTRACT_VERSION = "neda-v4-multitrace-learner-v1"
NEDA_TOKEN_ABLATION_METHODS = (
    "neda_no_position",
    "neda_token_only",
)
ALLOWED_CREDIT_CONTRACTS = {
    "neda-v4-online-credit-v1",
    "neda-v4-method-credit-v1",
    "neda-v4-one-step-credit-v1",
    "neda-joint-credit-v1",
    "neda-v2-joint-credit-v1",
}


def distributed_step_plan(
    n_rows: int, world_size: int, accumulation: int, epochs: int
) -> Dict[str, int]:
    """Return the synchronized step plan for complete-row data parallelism."""

    values = {
        "n_rows": int(n_rows),
        "world_size": int(world_size),
        "accumulation": int(accumulation),
        "epochs": int(epochs),
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError("distributed step-plan inputs must all be positive")
    local = (values["n_rows"] + values["world_size"] - 1) // values["world_size"]
    updates = (local + values["accumulation"] - 1) // values["accumulation"]
    return {
        **values,
        "local_microbatches_per_epoch": local,
        "optimizer_steps_per_epoch": updates,
        "expected_optimizer_steps": updates * values["epochs"],
        "padded_global_microbatches_per_epoch": local * values["world_size"],
    }


def pad_native_rows_for_distributed(
    rows: List[Dict[str, Any]], world_size: int, accumulation: int
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Append authenticated zero-credit rows instead of active sample repeats.

    Accelerate's default ``even_batches=True`` loops back to the beginning of
    a map-style dataset when its length is not divisible by the process count.
    Those implicit repeats would give a few real commitments extra policy
    weight.  We instead pad prospectively to one complete distributed
    accumulation quantum.  Padding rows retain a valid replay coordinate for
    collective symmetry but have exactly zero token and position credit.
    """

    world_size = int(world_size)
    accumulation = int(accumulation)
    if not rows:
        raise ValueError("distributed native-row padding requires data")
    if world_size <= 0 or accumulation <= 0:
        raise ValueError("distributed padding dimensions must be positive")
    quantum = world_size * accumulation
    result = [copy.deepcopy(dict(row)) for row in rows]
    for row in result:
        row["is_distributed_padding"] = False
    padding = (-len(result)) % quantum
    # Prefer short replay coordinates to minimize the cost of zero-gradient
    # collectives.  The stable index tie-break makes the receipt reproducible.
    sources = sorted(
        range(len(result)),
        key=lambda index: (
            len(result[index].get("extended_input_ids", [])),
            str(result[index].get("sample_id", "")),
            str(result[index].get("source", "")),
            int(result[index].get("round_id", -1)),
            index,
        ),
    )
    for padding_index in range(padding):
        source_index = sources[padding_index % len(sources)]
        row = copy.deepcopy(result[source_index])
        row["sample_id"] = "{}::distributed-padding-{:04d}".format(
            row["sample_id"], padding_index
        )
        row["adv_map"] = [0.0] * len(row["adv_map"])
        if row.get("step_credit") is not None:
            row["step_credit"] = 0.0
        row["is_distributed_padding"] = True
        row["distributed_padding_source_index"] = int(source_index)
        result.append(row)
    if len(result) % quantum != 0:
        raise ValueError("distributed native-row padding arithmetic drift")
    return result, {
        "unpadded_rows": len(rows),
        "padding_rows": padding,
        "padded_rows": len(result),
        "world_size": world_size,
        "gradient_accumulation_steps": accumulation,
        "padding_quantum": quantum,
    }


def validate_action_replay_support(trace: Dict[str, Any]) -> bool:
    """Validate the extra support needed to replay a constrained Action."""

    validate_decision_trace(
        trace, require_logprobs=True, require_sampling=True, exact_replay=False
    )
    constraint = str(trace["sampling"].get("constraint", "none"))
    allowed_rows = trace.get("constraint_allowed_token_ids")
    if constraint == "none":
        if allowed_rows is not None and any(row is not None for row in allowed_rows):
            raise ValueError("unconstrained Action trace contains constrained support")
        return True
    if constraint != "trie":
        raise ValueError("unsupported multitrace Action constraint")
    if trace.get("decision_contract_version") != "neda-v4-decision-trace-v1":
        raise ValueError("constrained exact replay requires the V4 decision contract")
    if not isinstance(allowed_rows, list) or len(allowed_rows) != len(
        trace["response_ids"]
    ):
        raise ValueError("constrained exact replay requires recorded trie allowed-token rows")
    for token, row in zip(trace["response_ids"], allowed_rows):
        values = [int(value) for value in (row or [])]
        if not values or len(values) != len(set(values)) or int(token) not in values:
            raise ValueError("invalid recorded trie allowed-token row")
    return True


def _load_records(path: str, source: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("{} learner data is empty".format(source))
    result = []
    for record in records:
        if record.get("credit_trace_kind") != source:
            raise ValueError("{} learner record source drift".format(source))
        credit_contract = record.get("neda_credit_contract") or record.get(
            "method_credit_contract"
        )
        if credit_contract is None:
            raise ValueError("{} learner record lacks a registered credit contract".format(source))
        if credit_contract not in ALLOWED_CREDIT_CONTRACTS:
            raise ValueError(
                "{} learner record uses unregistered credit contract: {!r}".format(
                    source, credit_contract
                )
            )
        trace = record.get("decision_traces", {}).get(source)
        if not isinstance(trace, dict) or not trace.get("response_ids"):
            raise ValueError("{} learner record has no trace".format(source))
        if source == "action":
            validate_action_replay_support(trace)
        else:
            validate_decision_trace(
                trace, require_logprobs=True, require_sampling=True,
                exact_replay=False,
            )
        sampling = trace["sampling"]
        if (
            abs(float(sampling["temperature"]) - 1.0) > 1e-9
            or int(sampling["top_k"]) != 0
            or abs(float(sampling["top_p"]) - 1.0) > 1e-9
        ):
            raise ValueError("multitrace v1 requires untruncated sampling")
        if source == "action":
            if trace.get("scoring_layout") != EXACT_ACTION_SCORING_LAYOUT:
                raise ValueError("Action learner requires fixed full-duplicate AR trace")
            if int(trace.get("replay_width", 0)) < len(trace["response_ids"]):
                raise ValueError("Action replay width does not cover its tokens")
        elif trace.get("scoring_layout") is not None or trace.get("replay_width") is not None:
            raise ValueError("Thought learner requires native recorded diffusion layout")
        if source == "thought":
            native = trace.get("native_replay")
            if (
                not isinstance(native, dict)
                or native.get("contract_version")
                != NATIVE_THOUGHT_REPLAY_CONTRACT_VERSION
                or native.get("scoring_layout")
                != NATIVE_THOUGHT_SCORING_LAYOUT
            ):
                raise ValueError("Thought learner requires authenticated native replay state")
            if record.get("registered_method") in ("mapg", "dcolt", "neda"):
                if native.get("position_policy") not in (
                    "mapg_logit", "dcolt_upm"
                ):
                    raise ValueError("joint Thought learner lacks a position policy")
                position_trace = native.get("position_trace")
                if not isinstance(position_trace, list) or not position_trace:
                    raise ValueError("joint Thought learner lacks position decisions")
        adv_map = [float(value) for value in record.get("adv_map", [])]
        if len(adv_map) != len(trace["response_ids"]) or any(
            not math.isfinite(value) for value in adv_map
        ):
            raise ValueError("{} adv_map/trace alignment drift".format(source))
        result.append(
            {
                "record": record,
                "trace": trace,
                "adv_map": adv_map,
                "credit_contract": credit_contract,
            }
        )
    return result


def _position_ids(start_pos: int, response_width: int) -> List[int]:
    original = list(range(start_pos + response_width))
    return original + original[start_pos:]


def build_native_rows(
    thought_path: str,
    action_path: str,
    mask_id: int,
    sample_order_seed: int,
    thought_block_size: int,
) -> List[Dict[str, Any]]:
    rows = []
    for source, path, block_size in (
        ("thought", thought_path, int(thought_block_size)),
        ("action", action_path, 1),
    ):
        for item in _load_records(path, source):
            record, trace, credit = item["record"], item["trace"], item["adv_map"]
            policy_response_length = len(trace["response_ids"])
            native = trace.get("native_replay") if source == "thought" else None
            if native is not None:
                response_ids = list(native["response_ids"])
                response_length = len(response_ids)
                response_width = int(native["replay_width"])
                padding = response_width - response_length
                if padding < 0:
                    raise ValueError("Thought native replay width is shorter than its state")
                step_map = list(native["step_map"]) + [-1] * padding
                rollout_logp = list(native["behavior_logprobs"]) + [0.0] * padding
                native_optimization_mask = list(native["optimization_mask"])
                optimization_mask = native_optimization_mask + [False] * padding
                block_size = int(native["block_size"])
                attention_layout = str(native["scoring_layout"])
                expanded_credit = []
                credit_index = 0
                for enabled in native_optimization_mask:
                    if enabled:
                        expanded_credit.append(float(credit[credit_index]))
                        credit_index += 1
                    else:
                        expanded_credit.append(0.0)
                if credit_index != len(credit):
                    raise ValueError("Thought native replay/credit projection drift")
                expanded_credit.extend([0.0] * padding)
            else:
                response_ids = list(trace["response_ids"])
                response_length = len(response_ids)
                response_width = int(trace.get("replay_width", response_length))
                step_map = list(trace["step_map"]) + [-1] * (
                    response_width - response_length
                )
                rollout_logp = list(trace["behavior_logprobs"]) + [0.0] * (
                    response_width - response_length
                )
                optimization_mask = [
                    index < response_length for index in range(response_width)
                ]
                attention_layout = str(trace.get("scoring_layout") or "response-block-v1")
                expanded_credit = list(credit) + [0.0] * (
                    response_width - response_length
                )
            start_pos = len(trace["prefix_ids"])
            response = response_ids + [int(mask_id)] * (
                response_width - response_length
            )
            adv_map = expanded_credit
            if not (
                len(step_map)
                == len(rollout_logp)
                == len(optimization_mask)
                == len(adv_map)
                == response_width
            ):
                raise ValueError(
                    "{} native replay tensor-width drift".format(source)
                )
            recorded_allowed = trace.get("constraint_allowed_token_ids")
            if recorded_allowed is None:
                allowed = [None] * response_width
            else:
                allowed = [list(values) for values in recorded_allowed] + [None] * (
                    response_width - response_length
                )
            replay = exact_replay_rows(
                input_ids=list(trace["prefix_ids"]) + response,
                start_pos=start_pos,
                response_width=response_width,
                step_map=step_map,
                mask_id=int(mask_id),
                action_mask=optimization_mask,
            )
            if not replay:
                raise ValueError("{} trace produced no exact replay rows".format(source))
            for replay_row in replay:
                position_decision = None
                step_credit = None
                registered_method = str(
                    record.get("registered_method", "neda")
                )
                if source == "thought" and registered_method in (
                    "mapg",
                    "dcolt",
                    "neda",
                ) + NEDA_TOKEN_ABLATION_METHODS:
                    credit_map = {
                        int(key): float(value)
                        for key, value in record.get(
                            "step_credit_by_round", {}
                        ).items()
                    }
                    if int(replay_row["round_id"]) not in credit_map:
                        # K-boundary exact-path subsampling: this realized
                        # denoising row is authenticated but not selected for
                        # the StepMerge-style policy-gradient estimate.
                        continue
                    step_credit = credit_map[int(replay_row["round_id"])]
                    if registered_method in ("mapg", "dcolt", "neda"):
                        matches = [
                            dict(value)
                            for value in native["position_trace"]
                            if int(value["round_id"])
                            == int(replay_row["round_id"])
                        ]
                        if len(matches) != 1:
                            raise ValueError(
                                "joint replay row lacks one position decision"
                            )
                        position_decision = matches[0]
                        candidates = [
                            int(value)
                            for value in position_decision[
                                "candidate_positions"
                            ]
                        ]
                        selected_positions = [
                            int(value)
                            for value in position_decision[
                                "selected_positions"
                            ]
                        ]
                        if (
                            len(candidates) != len(set(candidates))
                            or len(selected_positions)
                            != len(set(selected_positions))
                            or any(
                                value not in candidates
                                for value in selected_positions
                            )
                        ):
                            raise ValueError(
                                "invalid recorded position support"
                            )
                        optimized = {
                            index - start_pos
                            for index, enabled in enumerate(
                                replay_row["prediction_mask"]
                            )
                            if enabled
                        }
                        if not optimized or not optimized.issubset(
                            set(selected_positions)
                        ):
                            raise ValueError(
                                "position decision does not cover optimized "
                                "commits"
                            )
                rows.append(
                    {
                        "sample_id": str(record["sample_id"]),
                        "source": source,
                        "registered_method": registered_method,
                        "credit_contract": item["credit_contract"],
                        "step_selection": record.get("step_selection"),
                        "round_id": int(replay_row["round_id"]),
                        "start_pos": start_pos,
                        "response_width": response_width,
                        "block_size": block_size,
                        "attention_layout": attention_layout,
                        "extended_input_ids": replay_row["extended_input_ids"],
                        "prediction_mask": replay_row["prediction_mask"],
                        "labels": replay_row["label_ids"],
                        "position_ids": _position_ids(start_pos, response_width),
                        "adv_map": adv_map,
                        "rollout_logp": [0.0] * start_pos + rollout_logp,
                        "constraint_allowed_token_ids": allowed,
                        "policy_response_length": policy_response_length,
                        "position_decision": position_decision,
                        "step_credit": step_credit,
                        "position_mask_index": (
                            [
                                int(value) == int(mask_id)
                                for value in replay_row["extended_input_ids"][
                                    start_pos : start_pos + response_width
                                ]
                            ]
                            if position_decision is not None else None
                        ),
                    }
                )
    rows.sort(
        key=lambda row: (
            stable_seed(
                int(sample_order_seed), row["sample_id"], row["source"],
                row["round_id"], "multitrace-order",
            ),
            row["sample_id"], row["source"], row["round_id"],
        )
    )
    return rows
