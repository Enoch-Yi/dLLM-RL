"""Pure contracts for the submission-scale NeDA credit benchmark.

This module intentionally has no torch/model dependency.  It is shared by the
GPU runner, login-node audits, and unit tests so the frozen anchor mapping can
be reconstructed without loading SDAR.
"""

import math
from typing import Any, Dict, List, Mapping, Sequence

from neda_repro import sha256_json, stable_seed


ALF_BASE_CONTRACT_VERSION = "neda-credit-alf-base-v1"
ALF_BRANCH_CONTRACT_VERSION = "neda-credit-alf-branch-v1"
ALF_BASE_GATE_CONTRACT_VERSION = "neda-credit-alf-base-gate-v1"
ALF_AUDIT_CONTRACT_VERSION = "neda-credit-alf-audit-v1"


def deterministic_categorical_choice(
    probabilities: Mapping[str, float], decision_seed: int
) -> Dict[str, Any]:
    """Draw reproducibly from a named finite categorical distribution.

    This pure helper is shared by the GPU sampler and offline receipt audit.
    Sorting by Action makes the draw invariant to environment/list ordering;
    deriving the unit variate from ``stable_seed`` avoids library RNG drift.
    """

    rows = sorted((str(action), float(probability)) for action, probability in probabilities.items())
    if not rows or any(not math.isfinite(value) or value < 0.0 for _, value in rows):
        raise ValueError("categorical probabilities must be finite and nonnegative")
    total = math.fsum(value for _, value in rows)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("categorical distribution has no positive mass")
    derived = stable_seed(int(decision_seed), "conditional-action-categorical-u31")
    unit = (float(derived) + 0.5) / float(2**31 - 1)
    threshold = unit * total
    cumulative = 0.0
    selected = rows[-1][0]
    for action, probability in rows:
        cumulative += probability
        if threshold < cumulative:
            selected = action
            break
    return {
        "decision_seed": int(decision_seed),
        "derived_u31": int(derived),
        "unit_interval_value": unit,
        "normalizer": total,
        "selected_action": selected,
    }


def quantile_anchor_turns(horizon: int, count: int = 4) -> List[Dict[str, Any]]:
    """Select distinct, endpoint-inclusive integer turn quantiles."""

    horizon = int(horizon)
    count = int(count)
    if count < 2 or horizon < count:
        raise ValueError(
            "credit benchmark requires {} distinct turns, horizon={}".format(
                count, horizon
            )
        )
    indices = [index * (horizon - 1) // (count - 1) for index in range(count)]
    if len(set(indices)) != count:
        raise ValueError("quantile anchor selection produced duplicate turns")
    return [
        {"stratum": "q{}of{}".format(index, count - 1), "turn_id": turn_id}
        for index, turn_id in enumerate(indices)
    ]


def build_credit_anchors(
    episodes: Sequence[Mapping[str, Any]], anchors_per_episode: int = 4
) -> List[Dict[str, Any]]:
    """Rebuild the immutable episode/quantile-to-anchor mapping."""

    rows = []
    for episode in episodes:
        for position in quantile_anchor_turns(
            len(episode["turns"]), anchors_per_episode
        ):
            turn_id = int(position["turn_id"])
            turn = episode["turns"][turn_id]
            anchor_id = "credit-anchor-{}".format(
                sha256_json(
                    [episode["episode_id"], position["stratum"], turn_id]
                )[:24]
            )
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "episode_id": str(episode["episode_id"]),
                    "game_id": str(episode["game_id"]),
                    "rollout_id": int(episode["rollout_id"]),
                    "stratum": str(position["stratum"]),
                    "turn_id": turn_id,
                    "state_sha256": str(turn["state_before"]["state_sha256"]),
                }
            )
    identifiers = [row["anchor_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate credit benchmark anchor IDs")
    return rows
