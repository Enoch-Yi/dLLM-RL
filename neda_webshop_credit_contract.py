#!/usr/bin/env python3
"""Pure WebShop credit-benchmark contracts shared by runner and auditor."""

from typing import Any, Dict, List, Mapping, Sequence

from neda_credit_benchmark import quantile_anchor_turns
from neda_repro import sha256_json


BASE_CONTRACT_VERSION = "neda-webshop-credit-base-v1"
BASE_GATE_CONTRACT_VERSION = "neda-webshop-credit-base-gate-v1"
BRANCH_CONTRACT_VERSION = "neda-webshop-credit-branch-v1"
AUDIT_CONTRACT_VERSION = "neda-webshop-credit-audit-v1"


def build_webshop_anchors(
    episodes: Sequence[Mapping[str, Any]], count: int
) -> List[Dict[str, Any]]:
    anchors: List[Dict[str, Any]] = []
    names = ("early", "middle", "late") if int(count) == 3 else tuple(
        "q{}".format(index) for index in range(int(count))
    )
    for episode in episodes:
        turns = list(episode.get("turns", []))
        selected = quantile_anchor_turns(len(turns), int(count))
        for stratum, position in zip(names, selected):
            turn_id = int(position["turn_id"])
            turn = turns[turn_id]
            anchor_id = "web-anchor-{}".format(
                sha256_json([episode["episode_id"], stratum, turn_id])[:20]
            )
            anchors.append(
                {
                    "anchor_id": anchor_id,
                    "episode_id": str(episode["episode_id"]),
                    "game_id": str(episode["game_id"]),
                    "webshop_task_id": int(episode["webshop_task_id"]),
                    "rollout_id": int(episode["rollout_id"]),
                    "stratum": str(stratum),
                    "turn_id": turn_id,
                    "state_sha256": str(turn["state_before"]["state_sha256"]),
                }
            )
    if len({row["anchor_id"] for row in anchors}) != len(anchors):
        raise ValueError("duplicate WebShop credit anchor IDs")
    return anchors
