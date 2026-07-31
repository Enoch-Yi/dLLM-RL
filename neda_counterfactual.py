"""Counterfactual turn-credit contracts for NeDA V4-E04/O01.

The module is deliberately torch/ALFWorld-free.  Login-node tests can therefore
validate state restoration, anchor selection, common-random-number (CRN) seed
pairing, and Monte-Carlo reference statistics before a GPU job is submitted.

The primary O01 estimand under the current agent interface is an Action-only
turn intervention.  The deployed prompt history stores the executed Action and
the resulting observation, not the model's hidden/free-form Thought.  A
Thought-only effect is consequently structural zero under this memory contract
and must not be presented as an empirically identified effect.
"""

from __future__ import division

import math
import random
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from neda_repro import canonical_game_id, sha256_json, stable_seed


COUNTERFACTUAL_CONTRACT_VERSION = "neda-counterfactual-turn-v1"
STATE_CONTRACT_VERSION = "neda-alfworld-state-v1"
CRN_CONTRACT_VERSION = "neda-counterfactual-crn-v1"
MEMORY_CONTRACT = "executed-action-observation-only-v1"
ANCHOR_STRATA = ("early", "middle", "late")


def canonical_observation(value: Any) -> str:
    """Canonicalize only transport-level newline/edge whitespace differences."""

    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def canonical_admissible(commands: Iterable[Any]) -> List[str]:
    """Represent an admissible-command *set* deterministically."""

    values = [str(command).strip() for command in commands]
    if len(values) != len(set(values)):
        raise ValueError("admissible command list contains duplicates")
    return sorted(values)


def make_state_record(
    observation: Any,
    admissible_commands: Iterable[Any],
    cumulative_reward: float = 0.0,
    done: bool = False,
) -> Dict[str, Any]:
    payload = {
        "observation": canonical_observation(observation),
        "admissible_commands": canonical_admissible(admissible_commands),
        "cumulative_reward": float(cumulative_reward),
        "done": bool(done),
    }
    result = {"contract_version": STATE_CONTRACT_VERSION, **payload}
    result["state_sha256"] = sha256_json(payload)
    return result


def validate_state_record(state: Mapping[str, Any]) -> bool:
    if state.get("contract_version") != STATE_CONTRACT_VERSION:
        raise ValueError("unsupported state contract: {!r}".format(state.get("contract_version")))
    expected = make_state_record(
        state.get("observation", ""),
        state.get("admissible_commands", []),
        state.get("cumulative_reward", 0.0),
        state.get("done", False),
    )
    if state.get("state_sha256") != expected["state_sha256"]:
        raise ValueError("state_sha256 does not match state payload")
    return True


def assert_state_match(
    expected: Mapping[str, Any], actual: Mapping[str, Any], context: str
) -> None:
    validate_state_record(expected)
    validate_state_record(actual)
    if expected["state_sha256"] != actual["state_sha256"]:
        fields = []
        for key in ("observation", "admissible_commands", "cumulative_reward", "done"):
            if expected.get(key) != actual.get(key):
                fields.append(key)
        raise ValueError(
            "state prefix mismatch at {} (fields={}; expected={}; actual={})".format(
                context,
                ",".join(fields) or "unknown",
                expected["state_sha256"],
                actual["state_sha256"],
            )
        )


def anchor_turns(horizon: int) -> List[Dict[str, Any]]:
    """Select distinct early/middle/late turns for an eligible episode.

    O01 pre-registers three anchors per episode.  Episodes shorter than three
    turns cannot provide three distinct positions and fail closed rather than
    silently counting the same decision multiple times.
    """

    horizon = int(horizon)
    if horizon < 3:
        raise ValueError("episode requires at least three turns for early/middle/late anchors")
    indices = (0, horizon // 2, horizon - 1)
    if len(set(indices)) != 3:
        raise ValueError("anchor selection did not produce three distinct turns")
    return [
        {"stratum": stratum, "turn_id": int(turn_id)}
        for stratum, turn_id in zip(ANCHOR_STRATA, indices)
    ]


def make_anchor_id(episode_id: str, stratum: str, turn_id: int) -> str:
    if stratum not in ANCHOR_STRATA:
        raise ValueError("unknown anchor stratum: {}".format(stratum))
    return "anchor-{}".format(
        sha256_json([str(episode_id), str(stratum), int(turn_id)])[:20]
    )


def build_anchor_records(base_episodes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    anchors: List[Dict[str, Any]] = []
    for episode in base_episodes:
        validate_base_episode(episode)
        turns = list(episode.get("turns", []))
        for position in anchor_turns(len(turns)):
            turn_id = position["turn_id"]
            turn = turns[turn_id]
            if int(turn.get("turn_id", -1)) != turn_id:
                raise ValueError("base episode turn IDs are not contiguous")
            anchor_id = make_anchor_id(episode["episode_id"], position["stratum"], turn_id)
            anchors.append(
                {
                    "anchor_id": anchor_id,
                    "episode_id": str(episode["episode_id"]),
                    "game_id": str(episode["game_id"]),
                    "rollout_id": int(episode["rollout_id"]),
                    "stratum": position["stratum"],
                    "turn_id": turn_id,
                    "state_sha256": str(turn["state_before"]["state_sha256"]),
                }
            )
    ids = [row["anchor_id"] for row in anchors]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate anchor IDs")
    return anchors


def validate_base_episode(episode: Mapping[str, Any]) -> bool:
    required = ("game_id", "episode_id", "rollout_id", "initial_state", "turns")
    missing = [key for key in required if key not in episode]
    if missing:
        raise ValueError("base episode is missing {}".format(missing))
    validate_state_record(episode["initial_state"])
    previous = episode["initial_state"]
    turns = list(episode["turns"])
    for turn_id, turn in enumerate(turns):
        if int(turn.get("turn_id", -1)) != turn_id:
            raise ValueError("base episode turn IDs are not contiguous")
        turn_missing = [
            key
            for key in ("state_before", "state_after", "raw_action", "executed_action")
            if key not in turn
        ]
        if turn_missing:
            raise ValueError("base turn is missing {}".format(turn_missing))
        assert_state_match(previous, turn["state_before"], "base-turn-{}-before".format(turn_id))
        validate_state_record(turn["state_after"])
        if str(turn.get("raw_action")) != str(turn.get("executed_action")):
            raise ValueError("counterfactual base requires raw Action == executed Action")
        previous = turn["state_after"]
    if "horizon" in episode and int(episode["horizon"]) != len(turns):
        raise ValueError("base episode horizon does not match turn count")
    if turns and "return" in episode:
        if abs(float(episode["return"]) - float(turns[-1]["state_after"]["cumulative_reward"])) > 1e-12:
            raise ValueError("base episode return does not match terminal state")
    return True


def select_retest_anchor_ids(
    anchor_ids: Sequence[str], fraction: float, selection_seed: int
) -> List[str]:
    """Choose exactly ceil(fraction*N) anchors independent of input order."""

    fraction = float(fraction)
    if not (0.0 <= fraction <= 1.0):
        raise ValueError("retest fraction must be in [0,1]")
    unique = sorted(set(str(value) for value in anchor_ids))
    if len(unique) != len(anchor_ids):
        raise ValueError("retest selection received duplicate anchor IDs")
    n_select = int(math.ceil(fraction * len(unique))) if unique else 0
    ranked = sorted(
        unique,
        key=lambda anchor_id: sha256_json([int(selection_seed), anchor_id, "retest"]),
    )
    return sorted(ranked[:n_select])


def make_crn_schedule(
    branch_seed: int,
    anchor_id: str,
    repeat_id: int,
    sample_id: int,
    max_future_turns: int,
) -> Dict[str, Any]:
    """Create branch-label-free seeds shared by an original/alternative pair."""

    pair_seed = stable_seed(
        int(branch_seed), str(anchor_id), int(repeat_id), int(sample_id), "pair"
    )
    result = {
        "contract_version": CRN_CONTRACT_VERSION,
        "branch_seed": int(branch_seed),
        "anchor_id": str(anchor_id),
        "repeat_id": int(repeat_id),
        "sample_id": int(sample_id),
        "pair_seed": int(pair_seed),
        "env_reset_seed": stable_seed(pair_seed, "env-reset"),
        "intervention_seed": stable_seed(pair_seed, "intervention"),
        "continuation_decision_seeds": [
            stable_seed(pair_seed, offset, "continuation")
            for offset in range(max(0, int(max_future_turns)))
        ],
    }
    result["schedule_sha256"] = sha256_json(
        {key: value for key, value in result.items() if key != "schedule_sha256"}
    )
    return result


def validate_crn_schedule(schedule: Mapping[str, Any]) -> bool:
    if schedule.get("contract_version") != CRN_CONTRACT_VERSION:
        raise ValueError("unsupported CRN contract")
    expected = make_crn_schedule(
        schedule["branch_seed"],
        schedule["anchor_id"],
        schedule["repeat_id"],
        schedule["sample_id"],
        len(schedule.get("continuation_decision_seeds", [])),
    )
    if dict(schedule) != expected:
        raise ValueError("stored CRN schedule is inconsistent")
    return True


def choose_uniform_alternative_action(
    admissible_commands: Sequence[str], original_action: str, intervention_seed: int
) -> str:
    candidates = [
        command
        for command in canonical_admissible(admissible_commands)
        if command != str(original_action)
    ]
    if not candidates:
        raise ValueError("anchor has no admissible Action alternative")
    return candidates[random.Random(int(intervention_seed)).randrange(len(candidates))]


def _first_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0]
    if hasattr(value, "shape") and getattr(value, "shape", ()):
        return value[0]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _first_observation(value: Any) -> str:
    return str(_first_scalar(value))


def _admissible_from_info(info: Mapping[str, Any]) -> List[str]:
    value = info.get("admissible_commands", [])
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [str(command) for command in value]


def _game_id_from_info(info: Mapping[str, Any]) -> Any:
    for key in ("extra.gamefile", "gamefile"):
        if key not in info:
            continue
        value = _first_scalar(info[key])
        if value:
            try:
                return canonical_game_id(str(value))
            except ValueError:
                return str(value)
    return None


def replay_environment_prefix(
    env: Any,
    episode: Mapping[str, Any],
    anchor_turn: int,
    env_reset_seed: int,
    seed_callback: Callable[[int], None],
    max_history: Any = None,
) -> Dict[str, Any]:
    """Restore an ALFWorld state by exact Action replay from episode start.

    ``env`` only needs batched ``reset`` and ``step`` methods, so deterministic
    mock environments can exercise this contract on a login node.  The caller
    receives live observation/info/history alongside a serializable PASS audit.
    Any observation, admissible-set, reward, done, game, or action-prefix drift
    raises immediately.
    """

    turns = list(episode.get("turns", []))
    anchor_turn = int(anchor_turn)
    if not (0 <= anchor_turn < len(turns)):
        raise ValueError("anchor turn is outside the base episode")
    seed_callback(int(env_reset_seed))
    observations, info = env.reset()
    observation = _first_observation(observations)
    cumulative_reward = 0.0
    done = False
    actual = make_state_record(
        observation, _admissible_from_info(info), cumulative_reward, done
    )
    assert_state_match(episode["initial_state"], actual, "reset")
    actual_game_id = _game_id_from_info(info)
    if actual_game_id is not None and actual_game_id != str(episode["game_id"]):
        raise ValueError(
            "prefix replay reset the wrong game: expected {}, got {}".format(
                episode["game_id"], actual_game_id
            )
        )
    history = ["Observation: {}".format(canonical_observation(observation))]
    executed_actions: List[str] = []
    state_hashes = [actual["state_sha256"]]
    for turn_id in range(anchor_turn):
        turn = turns[turn_id]
        if int(turn.get("turn_id", -1)) != turn_id:
            raise ValueError("base prefix turn IDs are not contiguous")
        assert_state_match(turn["state_before"], actual, "turn-{}-before".format(turn_id))
        action = str(turn["executed_action"])
        if actual["done"]:
            raise ValueError("base prefix continues after a terminal state")
        observations, rewards, dones, info = env.step([action])
        observation = _first_observation(observations)
        cumulative_reward = float(_first_scalar(rewards))
        done = bool(_first_scalar(dones))
        actual = make_state_record(
            observation, _admissible_from_info(info), cumulative_reward, done
        )
        assert_state_match(turn["state_after"], actual, "turn-{}-after".format(turn_id))
        executed_actions.append(action)
        state_hashes.append(actual["state_sha256"])
        history.append("Thought/Action: {}".format(action))
        history.append("Observation: {}".format(canonical_observation(observation)))
        if max_history is not None and len(history) > int(max_history):
            if int(max_history) < 3:
                raise ValueError("max_history must retain the initial observation and one turn")
            history = history[:1] + history[-(int(max_history) - 1) :]
    assert_state_match(
        turns[anchor_turn]["state_before"], actual, "anchor-{}".format(anchor_turn)
    )
    audit = {
        "status": "PASS",
        "env_reset_seed": int(env_reset_seed),
        "n_replayed_transitions": anchor_turn,
        "executed_actions_sha256": sha256_json(executed_actions),
        "state_path_sha256": sha256_json(state_hashes),
        "anchor_state_sha256": actual["state_sha256"],
    }
    return {
        "observation": observation,
        "info": info,
        "cumulative_reward": cumulative_reward,
        "done": done,
        "history": history,
        "state": actual,
        "audit": audit,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values))


def summarize_paired_effects(pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize paired Q(original)-Q(alternative) Monte-Carlo samples."""

    effects: List[float] = []
    original_returns: List[float] = []
    alternative_returns: List[float] = []
    for pair in pairs:
        schedule = pair.get("crn_schedule")
        if not isinstance(schedule, Mapping):
            raise ValueError("counterfactual pair is missing crn_schedule")
        validate_crn_schedule(schedule)
        original = pair.get("original", {})
        alternative = pair.get("alternative", {})
        for branch_name, branch in (("original", original), ("alternative", alternative)):
            if branch.get("crn_schedule_sha256") != schedule["schedule_sha256"]:
                raise ValueError("{} branch does not use the paired CRN schedule".format(branch_name))
        original_return = float(original["return"])
        alternative_return = float(alternative["return"])
        effect = original_return - alternative_return
        stored_effect = pair.get("paired_effect")
        if stored_effect is not None and abs(float(stored_effect) - effect) > 1e-12:
            raise ValueError("stored paired effect is inconsistent with branch returns")
        effects.append(effect)
        original_returns.append(original_return)
        alternative_returns.append(alternative_return)
    if not effects:
        raise ValueError("reference requires at least one valid CRN pair")
    mean_effect = _mean(effects)
    if len(effects) > 1:
        variance = sum((value - mean_effect) ** 2 for value in effects) / (len(effects) - 1)
        standard_deviation = math.sqrt(max(0.0, variance))
        standard_error = standard_deviation / math.sqrt(len(effects))
    else:
        standard_deviation = None
        standard_error = None
    return {
        "estimand": "mean_paired_return_difference",
        "n_pairs": len(effects),
        "effective_sample_size": float(len(effects)),
        "original_mean_return": _mean(original_returns),
        "alternative_mean_return": _mean(alternative_returns),
        "effect_samples": effects,
        "mean_effect": mean_effect,
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "normal_95ci": (
            None
            if standard_error is None
            else [mean_effect - 1.96 * standard_error, mean_effect + 1.96 * standard_error]
        ),
    }


def _average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(float(value) for value in values), key=lambda pair: pair[1])
    ranks = [0.0] * len(indexed)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average = (start + 1 + end) / 2.0
        for offset in range(start, end):
            ranks[indexed[offset][0]] = average
        start = end
    return ranks


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> Any:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = _mean(left), _mean(right)
    numerator = sum(
        (float(x) - left_mean) * (float(y) - right_mean)
        for x, y in zip(left, right)
    )
    left_ss = sum((float(value) - left_mean) ** 2 for value in left)
    right_ss = sum((float(value) - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_ss * right_ss)
    return None if denominator == 0.0 else numerator / denominator


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> Any:
    if len(left) != len(right) or len(left) < 2:
        return None
    return pearson_correlation(_average_ranks(left), _average_ranks(right))


def kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> Any:
    if len(left) != len(right) or len(left) < 2:
        return None
    concordant = discordant = ties_left = ties_right = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            dx = (left[first] > left[second]) - (left[first] < left[second])
            dy = (right[first] > right[second]) - (right[first] < right[second])
            if dx == 0 and dy == 0:
                ties_left += 1
                ties_right += 1
            elif dx == 0:
                ties_left += 1
            elif dy == 0:
                ties_right += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_left)
        * (concordant + discordant + ties_right)
    )
    return None if denominator == 0.0 else (concordant - discordant) / denominator


def sign_agreement(left: Sequence[float], right: Sequence[float], tolerance: float = 0.0) -> Any:
    if len(left) != len(right) or not left:
        return None

    def sign(value: float) -> int:
        value = float(value)
        if abs(value) <= tolerance:
            return 0
        return 1 if value > 0 else -1

    return sum(sign(x) == sign(y) for x, y in zip(left, right)) / float(len(left))


def summarize_test_retest(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compute reliability from independent repeat-0/repeat-1 references."""

    anchor_ids: List[str] = []
    first: List[float] = []
    second: List[float] = []
    for row in sorted(rows, key=lambda value: str(value["anchor_id"])):
        repeats = row.get("repeat_references", {})
        if "0" not in repeats or "1" not in repeats:
            continue
        anchor_ids.append(str(row["anchor_id"]))
        first.append(float(repeats["0"]["mean_effect"]))
        second.append(float(repeats["1"]["mean_effect"]))
    return {
        "n_anchors": len(anchor_ids),
        "anchor_ids": anchor_ids,
        "spearman": spearman_correlation(first, second),
        "kendall_tau_b": kendall_tau_b(first, second),
        "sign_agreement": sign_agreement(first, second),
        "repeat_0_effects": first,
        "repeat_1_effects": second,
    }


def branch_reproducibility_fingerprint(branch: Mapping[str, Any]) -> str:
    """Hash scientific branch outputs while excluding runtime/latency metadata."""

    payload = {
        "branch_role": branch.get("branch_role"),
        "anchor_action": branch.get("anchor_action"),
        "return": float(branch.get("return", 0.0)),
        "success": bool(branch.get("success", False)),
        "final_state_sha256": branch.get("final_state_sha256"),
        "trajectory": [
            {
                "turn_offset": int(turn.get("turn_offset", -1)),
                "decision_seed": turn.get("decision_seed"),
                "generation": turn.get("generation"),
                "raw_action": turn.get("raw_action"),
                "executed_action": turn.get("executed_action"),
                "state_after_sha256": turn.get("state_after_sha256"),
                "reward": float(turn.get("reward", 0.0)),
                "done": bool(turn.get("done", False)),
            }
            for turn in branch.get("trajectory", [])
        ],
    }
    return sha256_json(payload)
