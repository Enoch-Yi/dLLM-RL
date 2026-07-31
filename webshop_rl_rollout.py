#!/usr/bin/env python3
"""Exact, grouped WebShop rollouts for iterative NeDA V4 optimization.

Each invocation owns one or more frozen WebShop tasks and samples ``k``
trajectories per task from the current behavior checkpoint.  The output uses
the same trace-v2 row contract consumed by the ALFWorld exact-path learners,
while preserving WebShop task, state, progress, and Action-space identities.
"""

import argparse
import json
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from neda_data_contract import TRACE_CONTRACT_VERSION, validate_rollout_record
from neda_repro import sha256_file, sha256_json
from neda_webshop_env import BACKEND_ACTION_MAP_CONTRACT, canonical_text


JOINT_METHODS = ("mapg", "dcolt", "neda")
POSITION_POLICY_BY_METHOD = {
    "mapg": "mapg_logit",
    "dcolt": "dcolt_upm",
    "neda": "mapg_logit",
}


ROLLOUT_CONTRACT_VERSION = "neda-v4-webshop-online-rollout-v1"


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(value: Any, path: str) -> None:
    path = os.path.realpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def frozen_task_ids(path: str, split_name: str, offset: int, count: int) -> List[int]:
    manifest = _load_json(path)
    if manifest.get("contract_version") != "neda-webshop-v4-splits-v1":
        raise ValueError("unsupported frozen WebShop split manifest")
    split = manifest.get("splits", {}).get(split_name, {})
    values = [int(value) for value in split.get("task_ids", [])]
    if len(values) != int(split.get("n_tasks", -1)) or len(values) != len(set(values)):
        raise ValueError("frozen WebShop split count/uniqueness drift")
    if split.get("task_ids_sha256") != sha256_json(values):
        raise ValueError("frozen WebShop task ID SHA drift")
    offset, count = int(offset), int(count)
    if offset < 0 or count <= 0 or offset + count > len(values):
        raise ValueError("requested WebShop task window is outside the frozen split")
    return values[offset : offset + count]


def registered_seed_row(path: str, train_seed: int) -> Dict[str, Any]:
    manifest = _load_json(path)
    if manifest.get("contract_version") != "neda-seeds-v1":
        raise ValueError("unsupported NeDA seed registry")
    matches = [
        dict(row)
        for row in manifest.get("phases", {}).get("submission", [])
        if int(row.get("train_seed", -1)) == int(train_seed)
    ]
    if len(matches) != 1:
        raise ValueError("WebShop rollout train seed is not uniquely registered")
    return matches[0]


def episodes_to_rows(
    episodes: Sequence[Mapping[str, Any]],
    model_identity_sha256: str,
    task_schedule_seed: int,
    rollout_seed: int,
    task_offset: int,
) -> List[Dict[str, Any]]:
    if not episodes:
        raise ValueError("WebShop rollout episode set is empty")
    by_task: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_task[int(episode["webshop_task_id"])].append(episode)
    rows: List[Dict[str, Any]] = []
    for task_id, selected in sorted(by_task.items()):
        selected = sorted(selected, key=lambda value: int(value["rollout_id"]))
        returns = [float(value["return"]) for value in selected]
        mean = statistics.mean(returns)
        scale = statistics.pstdev(returns) if len(returns) > 1 else 0.0
        group_id = "web-group-{}".format(
            sha256_json([task_id, int(task_schedule_seed), int(task_offset)])[:20]
        )
        for episode, outcome in zip(selected, returns):
            advantage = (outcome - mean) / (scale + 1e-6) if scale > 0.0 else 0.0
            horizon = len(episode.get("turns", []))
            if horizon <= 0 or horizon != int(episode.get("horizon", -1)):
                raise ValueError("WebShop episode horizon is empty or inconsistent")
            episode_id = "web-episode-{}".format(
                sha256_json(
                    [group_id, int(episode["rollout_id"]), int(rollout_seed)]
                )[:20]
            )
            for turn_id, turn in enumerate(episode["turns"]):
                if int(turn.get("turn_id", -1)) != turn_id:
                    raise ValueError("WebShop episode turns are not contiguous")
                state = turn["state_before"]
                actions = list(turn["action_contract"]["actions"])
                record = {
                    "contract_version": TRACE_CONTRACT_VERSION,
                    "environment": "webshop",
                    "prompt": str(turn["prompt"]),
                    "response": str(turn["generation"]),
                    "generation": str(turn["generation"]),
                    "prompt_ids": [int(value) for value in turn["prompt_ids"]],
                    "response_ids": [int(value) for value in turn["response_ids"]],
                    "reward": float(advantage),
                    "group_advantage": float(advantage),
                    "turn_reward": float(turn["turn_reward"]),
                    "episode_return": float(outcome),
                    "game_id": "webshop-fixed-{}".format(task_id),
                    "webshop_task_id": int(task_id),
                    "group_id": group_id,
                    "episode_id": episode_id,
                    "rollout_id": int(episode["rollout_id"]),
                    "turn_id": int(turn_id),
                    "episode_horizon": int(horizon),
                    "sample_id": "web-turn-{}".format(
                        sha256_json([episode_id, turn_id])[:20]
                    ),
                    "game_seed": int(task_schedule_seed),
                    "rollout_seed": int(rollout_seed),
                    "decision_seed": int(turn["decision_seed"]),
                    "model_identity_sha256": str(model_identity_sha256),
                    "raw_action": str(turn["raw_action"]),
                    "executed_action": str(turn["executed_action"]),
                    "backend_action": str(turn["backend_action"]),
                    "action_transform": "trie",
                    "snap_score": 1.0,
                    "sent_is_legal": True,
                    "decision_traces": dict(turn["decision_traces"]),
                    "state_before": {
                        "contract_version": "neda-online-credit-state-v1",
                        "environment": "webshop",
                        "admissible_commands": actions,
                        "cumulative_reward": float(state["progress"]),
                        "webshop_state_sha256": str(state["state_sha256"]),
                        "action_space_sha256": str(
                            turn["action_contract"]["action_space_sha256"]
                        ),
                    },
                }
                if record["raw_action"] != record["executed_action"]:
                    raise ValueError("WebShop exact rollout requires raw Action == executed Action")
                if canonical_text(record["backend_action"]) != record["executed_action"]:
                    raise ValueError(
                        "WebShop exact rollout requires canonical/backend Action equivalence"
                    )
                action_contract = turn["action_contract"]
                if (
                    action_contract.get("backend_action_map_contract")
                    != BACKEND_ACTION_MAP_CONTRACT
                ):
                    raise ValueError("WebShop exact rollout backend Action map contract drift")
                backend_map = action_contract.get("backend_action_map", {})
                if backend_map.get(record["executed_action"]) != record["backend_action"]:
                    raise ValueError("WebShop exact rollout backend Action map drift")
                validate_rollout_record(record, require_exact_trace=True)
                rows.append(record)
    rows.sort(
        key=lambda value: (
            int(value["webshop_task_id"]),
            int(value["rollout_id"]),
            int(value["turn_id"]),
        )
    )
    return rows


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--model_dir", required=True)
    result.add_argument("--prompt_json", required=True)
    result.add_argument("--split_manifest", required=True)
    result.add_argument("--split_name", default="train")
    result.add_argument("--seed_manifest", required=True)
    result.add_argument("--train_seed", type=int, required=True)
    result.add_argument("--agentboard_root", required=True)
    result.add_argument("--web_url", default="http://127.0.0.1:3000")
    result.add_argument("--num_games", type=int, default=1)
    result.add_argument("--k", type=int, default=4)
    result.add_argument("--offset", type=int, default=0)
    result.add_argument("--max_steps", type=int, default=20)
    result.add_argument("--max_history", type=int, default=24)
    result.add_argument("--gen_length", type=int, default=64)
    result.add_argument("--action_gen_length", type=int, default=64)
    result.add_argument("--block_length", type=int, default=4)
    result.add_argument("--denoising_steps", type=int, default=4)
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--thought_order", choices=("ao", "ar"), default="ao")
    result.add_argument("--action_order", choices=("ao", "ar"), default="ar")
    result.add_argument("--action_grammar", choices=("none", "trie"), default="trie")
    result.add_argument("--rl_method", choices=JOINT_METHODS, required=True)
    result.add_argument("--position_temperature", type=float, default=0.5)
    result.add_argument("--dcolt_head_path")
    result.add_argument("--neda_credit_boundaries", type=int, default=4)
    result.add_argument("--task_schedule_seed", type=int, default=92001)
    result.add_argument("--rollout_seed", type=int)
    result.add_argument("--out", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    # Keep the pure contract/conversion functions importable on login nodes;
    # model/Torch imports are needed only by the actual GPU rollout process.
    from webshop_credit_benchmark import WebRuntime, collect_base_episode

    if args.action_order != "ar" or args.action_grammar != "trie":
        raise ValueError("formal WebShop rollouts require canonical AR+Trie Action")
    if int(args.k) < 2:
        raise ValueError("group-relative WebShop rollouts require k>=2")
    registered = registered_seed_row(args.seed_manifest, args.train_seed)
    expected_rollout_seed = int(registered["rollout_seed"])
    if args.rollout_seed is None:
        args.rollout_seed = expected_rollout_seed
    if int(args.rollout_seed) != expected_rollout_seed:
        raise ValueError("runtime WebShop rollout seed differs from the frozen registry")
    task_ids = frozen_task_ids(
        args.split_manifest, args.split_name, args.offset, args.num_games
    )
    args.base_rollout_seed = int(args.rollout_seed)
    runtime = WebRuntime(args)
    episodes = []
    for local_index, task_id in enumerate(task_ids):
        for rollout_id in range(int(args.k)):
            episode = collect_base_episode(
                runtime,
                args,
                int(task_id),
                int(args.offset) + local_index,
                rollout_id,
                validate_credit_anchor_coverage=False,
            )
            episodes.append(episode)
        print(
            "[web-rollout] task={} returns={}".format(
                task_id,
                [
                    round(float(value["return"]), 6)
                    for value in episodes[-int(args.k) :]
                ],
            ),
            flush=True,
        )
    rows = episodes_to_rows(
        episodes,
        runtime.model_identity["identity_sha256"],
        args.task_schedule_seed,
        args.rollout_seed,
        args.offset,
    )
    _atomic_json(rows, args.out)
    manifest = {
        "contract_version": TRACE_CONTRACT_VERSION,
        "rollout_contract_version": ROLLOUT_CONTRACT_VERSION,
        "status": "PASS",
        "environment": "webshop",
        "data_path": os.path.realpath(args.out),
        "data_sha256": sha256_file(args.out),
        "n_records": len(rows),
        "n_episodes": len(episodes),
        "environment_interactions": len(rows),
        "game_seed": int(args.task_schedule_seed),
        "rollout_seed": int(args.rollout_seed),
        "train_seed": int(args.train_seed),
        "offset": int(args.offset),
        "num_games": len(task_ids),
        "group_size": int(args.k),
        "max_steps": int(args.max_steps),
        "gen_length": int(args.gen_length),
        "action_gen_length": int(args.action_gen_length),
        "block_length": int(args.block_length),
        "denoising_steps": int(args.denoising_steps),
        "decision_decode": "two_stage",
        "thought_order": args.thought_order,
        "action_order": args.action_order,
        "action_grammar": args.action_grammar,
        "execution_policy": "raw",
        "temperature": float(args.temperature),
        "online_credit_context_contract": "neda-online-credit-state-v1",
        "rl_method": args.rl_method,
        "position_policy": POSITION_POLICY_BY_METHOD[args.rl_method],
        "position_temperature": float(args.position_temperature),
        "dcolt_head_path": (
            os.path.realpath(args.dcolt_head_path)
            if args.dcolt_head_path else None
        ),
        "dcolt_head_contract": (
            runtime.dcolt_head_metadata.get("contract_version")
            if runtime.dcolt_head_metadata else None
        ),
        "dcolt_head_sha256": (
            sha256_file(args.dcolt_head_path)
            if args.dcolt_head_path else None
        ),
        "neda_credit_boundaries": (
            int(args.neda_credit_boundaries)
            if args.rl_method == "neda" else None
        ),
        "task_ids": task_ids,
        "task_ids_sha256": sha256_json(task_ids),
        "split_manifest": os.path.realpath(args.split_manifest),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "seed_manifest": os.path.realpath(args.seed_manifest),
        "seed_manifest_sha256": sha256_file(args.seed_manifest),
        "sample_order_sha256": sha256_json([row["sample_id"] for row in rows]),
        "model_identity": runtime.model_identity,
    }
    _atomic_json(manifest, args.out + ".manifest.json")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
