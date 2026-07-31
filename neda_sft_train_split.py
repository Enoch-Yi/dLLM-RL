"""Deterministic ALFWorld train selection disjoint from Agentic-SFT games."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from neda_repro import canonical_game_id, sha256_json


CONTRACT_VERSION = "neda-sft-disjoint-alfworld-train-v1"


def select_sft_disjoint_games(
    game_files: Sequence[str],
    *,
    game_seed: int,
    offset: int,
    num_games: int,
    excluded_sorted_prefix_games: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """Select a deterministic slice after removing the SFT training prefix."""

    ordered = sorted(str(path) for path in game_files)
    excluded_count = int(excluded_sorted_prefix_games)
    if excluded_count < 0 or excluded_count >= len(ordered):
        raise ValueError("excluded SFT prefix must leave a nonempty RL pool")
    if int(offset) < 0 or int(num_games) <= 0:
        raise ValueError("offset and num_games must be nonnegative/positive")

    excluded = ordered[:excluded_count]
    eligible = ordered[excluded_count:]
    shuffled = list(eligible)
    random.Random(int(game_seed)).shuffle(shuffled)
    selected = shuffled[int(offset): int(offset) + int(num_games)]
    if len(selected) != int(num_games):
        raise ValueError("requested RL game slice exceeds the SFT-disjoint pool")

    excluded_ids = [canonical_game_id(path) for path in excluded]
    eligible_ids = [canonical_game_id(path) for path in eligible]
    selected_ids = [canonical_game_id(path) for path in selected]
    if set(selected_ids) & set(excluded_ids):
        raise ValueError("SFT/RL exact-game overlap")
    metadata: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "total_train_games": len(ordered),
        "excluded_sorted_prefix_games": excluded_count,
        "eligible_rl_games": len(eligible),
        "excluded_game_ids_sha256": sha256_json(excluded_ids),
        "eligible_game_ids_sha256": sha256_json(eligible_ids),
        "selected_game_ids_sha256": sha256_json(selected_ids),
    }
    return selected, metadata


def canonical_ids(paths: Sequence[str]) -> List[str]:
    return [canonical_game_id(Path(path)) for path in paths]
