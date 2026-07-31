#!/usr/bin/env python3
"""Frozen, shardable WebShop evaluation for NeDA V4."""

import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Sequence

from neda_repro import model_identities_match, sha256_file, sha256_json
from webshop_rl_rollout import frozen_task_ids


EVAL_CONTRACT_VERSION = "neda-v4-webshop-eval-v1"
MERGE_CONTRACT_VERSION = "neda-v4-webshop-eval-merge-v1"
JOINT_METHODS = ("mapg", "dcolt", "neda")
POSITION_POLICY_BY_METHOD = {
    "mapg": "mapg_logit",
    "dcolt": "dcolt_upm",
    "neda": "mapg_logit",
}


def _load(path: str) -> Any:
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


def merge_shards(paths: Sequence[str], expected_tasks: int) -> Dict[str, Any]:
    if not paths:
        raise ValueError("WebShop evaluation requires at least one shard")
    shards = [_load(path) for path in paths]
    for value in shards:
        if (
            value.get("contract_version") != EVAL_CONTRACT_VERSION
            or value.get("status") != "PASS"
        ):
            raise ValueError("WebShop evaluation shard is not PASS")
    reference = shards[0]
    invariants = (
        "split_name",
        "eval_seed",
        "samples_per_task",
        "max_steps",
        "action_gen_length",
        "temperature",
        "thought_order",
        "action_order",
        "action_grammar",
        "split_manifest_sha256",
        "rl_method",
        "position_policy",
        "position_temperature",
        "dcolt_head_sha256",
    )
    for value in shards[1:]:
        for field in invariants:
            if value.get(field) != reference.get(field):
                raise ValueError("WebShop eval shard protocol drift: {}".format(field))
        if not model_identities_match(value["model_identity"], reference["model_identity"]):
            raise ValueError("WebShop eval shard model identity drift")
    results = [dict(row) for value in shards for row in value.get("results", [])]
    results.sort(key=lambda row: int(row["webshop_task_id"]))
    task_ids = [int(row["webshop_task_id"]) for row in results]
    if len(results) != int(expected_tasks) or len(task_ids) != len(set(task_ids)):
        raise ValueError("WebShop eval task coverage/count drift")
    samples_per_task = int(reference["samples_per_task"])
    if any(len(row.get("samples", [])) != samples_per_task for row in results):
        raise ValueError("WebShop eval sample count drift")
    interactions = sum(
        int(sample["horizon"])
        for row in results
        for sample in row["samples"]
    )
    latencies = [
        float(sample["decision_latency_seconds"])
        for row in results
        for sample in row["samples"]
    ]
    receipts = [
        {"path": os.path.realpath(path), "sha256": sha256_file(path)}
        for path in paths
    ]
    output = {
        "contract_version": MERGE_CONTRACT_VERSION,
        "status": "PASS",
        "environment": "webshop",
        "split_name": reference["split_name"],
        "eval_seed": int(reference["eval_seed"]),
        "samples_per_task": samples_per_task,
        "num_tasks": len(results),
        "n_success_at_1": sum(
            bool(row.get("success_at_1", row["samples"][0]["success"]))
            for row in results
        ),
        "success_at_1": (
            sum(
                bool(row.get("success_at_1", row["samples"][0]["success"]))
                for row in results
            )
            / len(results)
        ),
        "n_success_at_k": sum(bool(row["success_at_k"]) for row in results),
        "success_at_k": (
            sum(bool(row["success_at_k"]) for row in results) / len(results)
        ),
        # Backward-compatible aliases: historically ``success_rate`` meant
        # task-level Success@k for the configured samples_per_task.
        "n_success": sum(bool(row["success_at_k"]) for row in results),
        "success_rate": (
            sum(bool(row["success_at_k"]) for row in results) / len(results)
        ),
        "score": sum(float(row["mean_score"]) for row in results) / len(results),
        "environment_interactions": interactions,
        "mean_decision_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else 0.0
        ),
        "task_ids": task_ids,
        "task_ids_sha256": sha256_json(task_ids),
        "model_identity": reference["model_identity"],
        "protocol": {field: reference.get(field) for field in invariants},
        "source_receipts": receipts,
        "source_set_sha256": sha256_json(receipts),
        "results": results,
    }
    output["artifact_sha256"] = sha256_json(output)
    return output


def run(args: argparse.Namespace) -> Dict[str, Any]:
    # Lazy import keeps merge/contract tests free of GPU/Torch dependencies.
    from webshop_credit_benchmark import WebRuntime, collect_base_episode

    task_ids = frozen_task_ids(
        args.split_manifest, args.split_name, args.offset, args.num_tasks
    )
    args.anchors_per_episode = 1
    args.base_rollout_seed = int(args.eval_seed)
    runtime = WebRuntime(args)
    results: List[Dict[str, Any]] = []
    for local_index, task_id in enumerate(task_ids):
        episodes = [
            collect_base_episode(
                runtime,
                args,
                int(task_id),
                int(args.offset) + local_index,
                sample_id,
                validate_credit_anchor_coverage=False,
            )
            for sample_id in range(int(args.samples_per_task))
        ]
        samples = []
        for episode in episodes:
            latency = sum(
                float(turn["decision_latency_seconds"])
                for turn in episode.get("turns", [])
            )
            samples.append(
                {
                    "sample_id": int(episode["rollout_id"]),
                    "score": float(episode["return"]),
                    "terminal_reward": float(episode["terminal_reward"]),
                    "success": bool(episode["success"]),
                    "horizon": int(episode["horizon"]),
                    "decision_latency_seconds": latency,
                    "episode_id": str(episode["episode_id"]),
                    "trajectory": episode["turns"],
                }
            )
        results.append(
            {
                "webshop_task_id": int(task_id),
                "game_id": "webshop-fixed-{}".format(task_id),
                "mean_score": sum(row["score"] for row in samples) / len(samples),
                "success_at_1": bool(samples[0]["success"]),
                "success_at_k": any(row["success"] for row in samples),
                "samples": samples,
            }
        )
        print(
            "[web-eval] task={} score={} success@{}={}".format(
                task_id,
                [round(row["score"], 6) for row in samples],
                args.samples_per_task,
                results[-1]["success_at_k"],
            ),
            flush=True,
        )
    return {
        "contract_version": EVAL_CONTRACT_VERSION,
        "status": "PASS",
        "environment": "webshop",
        "split_name": args.split_name,
        "split_manifest": os.path.realpath(args.split_manifest),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "task_offset": int(args.offset),
        "task_ids": task_ids,
        "task_ids_sha256": sha256_json(task_ids),
        "num_tasks": len(task_ids),
        "eval_seed": int(args.eval_seed),
        "samples_per_task": int(args.samples_per_task),
        "max_steps": int(args.max_steps),
        "action_gen_length": int(args.action_gen_length),
        "temperature": float(args.temperature),
        "thought_order": args.thought_order,
        "action_order": args.action_order,
        "action_grammar": args.action_grammar,
        "rl_method": args.rl_method,
        "position_policy": POSITION_POLICY_BY_METHOD[args.rl_method],
        "position_temperature": float(args.position_temperature),
        "dcolt_head_sha256": (
            sha256_file(args.dcolt_head_path)
            if args.dcolt_head_path else None
        ),
        "model_identity": runtime.model_identity,
        "results": results,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--model_dir", required=True)
    execute.add_argument("--prompt_json", required=True)
    execute.add_argument("--split_manifest", required=True)
    execute.add_argument("--split_name", choices=("credit_dev", "final_confirm"), required=True)
    execute.add_argument("--agentboard_root", required=True)
    execute.add_argument("--web_url", default="http://127.0.0.1:3000")
    execute.add_argument("--offset", type=int, required=True)
    execute.add_argument("--num_tasks", type=int, required=True)
    execute.add_argument("--samples_per_task", type=int, default=1)
    execute.add_argument("--eval_seed", type=int, required=True)
    execute.add_argument("--max_steps", type=int, default=20)
    execute.add_argument("--max_history", type=int, default=24)
    execute.add_argument("--gen_length", type=int, default=64)
    execute.add_argument("--action_gen_length", type=int, default=64)
    execute.add_argument("--block_length", type=int, default=4)
    execute.add_argument("--denoising_steps", type=int, default=4)
    execute.add_argument("--temperature", type=float, default=0.3)
    execute.add_argument("--thought_order", choices=("ao", "ar"), default="ao")
    execute.add_argument("--action_order", choices=("ao", "ar"), default="ar")
    execute.add_argument("--action_grammar", choices=("none", "trie"), default="trie")
    execute.add_argument("--rl_method", choices=JOINT_METHODS, required=True)
    execute.add_argument("--position_temperature", type=float, default=0.5)
    execute.add_argument("--dcolt_head_path")
    execute.add_argument("--neda_credit_boundaries", type=int, default=4)
    execute.add_argument("--out", required=True)
    combine = sub.add_parser("merge")
    combine.add_argument("--inputs", nargs="+", required=True)
    combine.add_argument("--expected_tasks", type=int, required=True)
    combine.add_argument("--out", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "run":
        if args.action_order != "ar" or args.action_grammar != "trie":
            raise ValueError("formal WebShop eval requires canonical AR+Trie Action")
        if int(args.samples_per_task) <= 0:
            raise ValueError("samples_per_task must be positive")
        value = run(args)
        value["artifact_sha256"] = sha256_json(value)
        _atomic_json(value, args.out)
    else:
        value = merge_shards(args.inputs, args.expected_tasks)
        _atomic_json(value, args.out)
    print(json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
