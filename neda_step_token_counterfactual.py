"""Pure contracts for NeDA V4-O02/O03 step/token interventions.

The primary local estimand is a controlled replacement effect.  At a recorded
Thought denoising state, one whole commit set (D) or one committed token (T) is
resampled from the frozen behavior policy with the realized token excluded.
All other realized Thought tokens are held fixed; the environment-facing
Action and subsequent trajectory are then regenerated.  This avoids claiming
an unidentified full diffusion-path likelihood or a unique causal oracle.
"""

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence

from neda_repro import sha256_json


STEP_TOKEN_CONTRACT_VERSION = "neda-step-token-counterfactual-v2"
REPLACEMENT_CONTRACT = "old-policy-excluding-realized-controlled-v1"


def _rank(seed: int, anchor_id: str, label: str, value: int) -> str:
    return sha256_json([int(seed), str(anchor_id), str(label), int(value)])


def make_coordinate_id(
    anchor_id: str, level: str, step_id: int, token_position: int = -1
) -> str:
    if level not in ("step", "token", "action_token"):
        raise ValueError("coordinate level must be step, token, or action_token")
    payload = [str(anchor_id), level, int(step_id)]
    if level in ("token", "action_token"):
        if int(token_position) < 0:
            raise ValueError("token coordinate requires a nonnegative position")
        payload.append(int(token_position))
    return "coord-{}".format(sha256_json(payload)[:24])


def complete_response_positions(
    prefix_length: int, response_length: int, block_size: int
) -> List[int]:
    """Return response offsets whose entire decoder block is recorded.

    Structural delimiter trimming can remove the tail of the final Thought
    block.  Such a block cannot reconstruct the exact pre-commit state and is
    therefore ineligible rather than silently padded.
    """

    prefix_length = int(prefix_length)
    response_length = int(response_length)
    block_size = int(block_size)
    if prefix_length < 0 or response_length < 0 or block_size <= 0:
        raise ValueError("invalid response geometry")
    recorded_end = prefix_length + response_length
    result = []
    for offset in range(response_length):
        absolute = prefix_length + offset
        block_end = (absolute // block_size + 1) * block_size
        if block_end <= recorded_end:
            result.append(offset)
    return result


def validate_thought_trace(trace: Mapping[str, Any]) -> bool:
    required = ("prefix_ids", "response_ids", "step_map", "behavior_logprobs")
    missing = [key for key in required if key not in trace]
    if missing:
        raise ValueError("Thought trace is missing {}".format(missing))
    length = len(trace["response_ids"])
    if length == 0:
        raise ValueError("Thought trace is empty")
    for key in ("step_map", "behavior_logprobs"):
        if len(trace[key]) != length:
            raise ValueError("Thought trace {} length mismatch".format(key))
    if any(int(value) < 0 for value in trace["step_map"]):
        raise ValueError("Thought trace contains uncommitted tokens")
    return True


def select_step_token_coordinates(
    trace: Mapping[str, Any],
    anchor_id: str,
    selection_seed: int,
    block_size: int = 4,
    n_steps: int = 2,
    tokens_per_step: int = 2,
) -> Dict[str, Any]:
    """Deterministically select exactly ``n_steps × tokens_per_step``.

    Eligible steps must be wholly inside one fully recorded block and contain
    enough co-committed tokens.  Any shortage fails closed.
    """

    validate_thought_trace(trace)
    n_steps = int(n_steps)
    tokens_per_step = int(tokens_per_step)
    if n_steps < 1 or tokens_per_step < 1:
        raise ValueError("selection counts must be positive")
    complete = set(
        complete_response_positions(
            len(trace["prefix_ids"]), len(trace["response_ids"]), block_size
        )
    )
    groups: Dict[int, List[int]] = defaultdict(list)
    for position, step_id in enumerate(trace["step_map"]):
        groups[int(step_id)].append(int(position))
    eligible = []
    for step_id, positions in groups.items():
        block_ids = {
            (len(trace["prefix_ids"]) + position) // int(block_size)
            for position in positions
        }
        if (
            len(positions) >= tokens_per_step
            and len(block_ids) == 1
            and all(position in complete for position in positions)
        ):
            eligible.append(step_id)
    if len(eligible) < n_steps:
        raise ValueError(
            "anchor {} has {} eligible steps; {} required".format(
                anchor_id, len(eligible), n_steps
            )
        )
    selected_steps = sorted(
        eligible,
        key=lambda value: _rank(selection_seed, anchor_id, "step", value),
    )[:n_steps]
    rows = []
    for step_id in selected_steps:
        positions = list(groups[step_id])
        selected_tokens = sorted(
            positions,
            key=lambda value: _rank(
                selection_seed, anchor_id, "token-{}".format(step_id), value
            ),
        )[:tokens_per_step]
        rows.append(
            {
                "step_id": int(step_id),
                "commit_positions": positions,
                "selected_token_positions": selected_tokens,
                "step_coordinate_id": make_coordinate_id(
                    anchor_id, "step", step_id
                ),
                "token_coordinate_ids": [
                    make_coordinate_id(anchor_id, "token", step_id, position)
                    for position in selected_tokens
                ],
            }
        )
    result = {
        "anchor_id": str(anchor_id),
        "selection_seed": int(selection_seed),
        "block_size": int(block_size),
        "n_steps": n_steps,
        "tokens_per_step": tokens_per_step,
        "n_eligible_steps": len(eligible),
        "steps": rows,
    }
    result["selection_sha256"] = sha256_json(result)
    return result


def replacement_positions(selection: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a selection into one step and N token coordinates per step."""

    rows: List[Dict[str, Any]] = []
    for step in selection.get("steps", []):
        rows.append(
            {
                "coordinate_id": step["step_coordinate_id"],
                "level": "step",
                "step_id": int(step["step_id"]),
                "positions": [int(value) for value in step["commit_positions"]],
            }
        )
        for coordinate_id, position in zip(
            step["token_coordinate_ids"], step["selected_token_positions"]
        ):
            rows.append(
                {
                    "coordinate_id": coordinate_id,
                    "level": "token",
                    "step_id": int(step["step_id"]),
                    "positions": [int(position)],
                }
            )
    return rows


def select_action_token_coordinates(
    trace: Mapping[str, Any],
    anchor_id: str,
    selection_seed: int,
    allowed_next,
    max_action_tokens: int = 2,
    min_action_tokens: int = 1,
) -> Dict[str, Any]:
    """Select identifiable AR Action-token interventions on the realized path.

    A position is eligible only when the frozen admissible-command Trie both
    accepts the realized token and offers at least one non-realized next token.
    The count is capped rather than forced to two because one-token commands
    (for example ``look``) can have only one identifiable branching position.
    Such anchors still provide a valid primary Action-token reference; an
    anchor with no branching position fails closed.
    """

    validate_thought_trace(trace)
    max_action_tokens = int(max_action_tokens)
    min_action_tokens = int(min_action_tokens)
    if max_action_tokens < 1 or not (1 <= min_action_tokens <= max_action_tokens):
        raise ValueError("invalid Action-token selection bounds")
    response = [int(value) for value in trace["response_ids"]]
    eligible = []
    prefix: List[int] = []
    for position, realized in enumerate(response):
        allowed = {int(value) for value in allowed_next(prefix)}
        if realized not in allowed:
            raise ValueError(
                "recorded Action token {} at position {} is outside the Trie".format(
                    realized, position
                )
            )
        alternatives = sorted(allowed - {realized})
        if alternatives:
            eligible.append(
                {
                    "position": int(position),
                    "realized_token_id": int(realized),
                    "n_alternative_token_ids": len(alternatives),
                    "alternative_token_ids_sha256": sha256_json(alternatives),
                }
            )
        prefix.append(realized)
    if len(eligible) < min_action_tokens:
        raise ValueError(
            "anchor {} has {} identifiable Action-token positions; {} required".format(
                anchor_id, len(eligible), min_action_tokens
            )
        )
    selected = sorted(
        eligible,
        key=lambda row: _rank(
            selection_seed, anchor_id, "action-token", int(row["position"])
        ),
    )[:max_action_tokens]
    rows = []
    for row in selected:
        position = int(row["position"])
        rows.append(
            {
                **row,
                "coordinate_id": make_coordinate_id(
                    anchor_id, "action_token", position, position
                ),
                "level": "action_token",
                # An AR Action token is also its own causal commit step.
                "step_id": position,
                "positions": [position],
            }
        )
    result = {
        "anchor_id": str(anchor_id),
        "selection_seed": int(selection_seed),
        "max_action_tokens": max_action_tokens,
        "min_action_tokens": min_action_tokens,
        "n_eligible_action_tokens": len(eligible),
        "coordinates": rows,
    }
    result["selection_sha256"] = sha256_json(result)
    return result
