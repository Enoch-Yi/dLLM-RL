#!/usr/bin/env python3
"""ALFWorld counterfactual branch runner for NeDA V4-E04/O01.

Two explicit stages keep expensive GPU work restartable:

``base``
    Collect stateful policy episodes.  Every turn stores the state before and
    after the executed Action, including exact observation/admissible hashes.

``branch``
    Restore each early/middle/late anchor by replaying the executed prefix from
    episode start, then run paired original/alternative continuations with a
    branch-label-free common-random-number seed schedule.

This is a Monte-Carlo reference harness, not a policy trainer and not a new
likelihood estimator.
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoTokenizer, GenerationConfig

from models import SDARForCausalLM
from neda_counterfactual import (
    COUNTERFACTUAL_CONTRACT_VERSION,
    MEMORY_CONTRACT,
    branch_reproducibility_fingerprint,
    build_anchor_records,
    canonical_observation,
    choose_uniform_alternative_action,
    make_crn_schedule,
    make_state_record,
    replay_environment_prefix,
    select_retest_anchor_ids,
    summarize_paired_effects,
)
from neda_data_contract import history_action, trim_generation_trace
from neda_freeze_splits import scan_games
from neda_repro import (
    build_model_identity,
    canonical_game_id,
    check_game_ids,
    load_split_manifest,
    model_identities_match,
    order_game_files_by_manifest,
    seed_everything,
    sha256_file,
    sha256_json,
    stable_seed,
)
from r002_alfworld import (
    build_action_trie,
    build_prompt,
    exact_ar_action_generate,
    two_stage_decision_decode,
)


BASE_ARTIFACT_KIND = "stateful-base-episodes"
BRANCH_ARTIFACT_KIND = "branch-results"
SEED_CONTRACT_VERSION = "neda-counterfactual-seeds-v1"


def atomic_json_dump(value: Mapping[str, Any], path: str) -> None:
    path = os.path.realpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_registered_seeds(args: argparse.Namespace) -> Dict[str, Any]:
    with open(args.counterfactual_seed_manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("contract_version") != SEED_CONTRACT_VERSION:
        raise ValueError("unsupported counterfactual seed manifest")
    registered = manifest.get(args.seed_phase)
    if args.seed_phase == "submission":
        if not isinstance(registered, list) or not (0 <= args.seed_replicate < len(registered)):
            raise ValueError("counterfactual submission seed replicate is out of range")
        registered = registered[args.seed_replicate]
    if not isinstance(registered, dict):
        raise ValueError("counterfactual seed phase is missing")
    actual = {
        "base_rollout_seed": int(args.base_rollout_seed),
        "branch_seed": int(args.branch_seed),
        "selection_seed": int(args.selection_seed),
    }
    expected = {key: int(registered[key]) for key in actual}
    if actual != expected:
        raise ValueError(
            "runtime counterfactual seeds differ from frozen manifest: actual={} expected={}".format(
                actual, expected
            )
        )
    return {
        "path": os.path.realpath(args.counterfactual_seed_manifest),
        "sha256": sha256_file(args.counterfactual_seed_manifest),
        "phase": args.seed_phase,
        "replicate": int(args.seed_replicate),
        "values": actual,
    }


def extract_goal(observation: str) -> str:
    for line in observation.split("\n"):
        if "your task is to:" in line.lower():
            return line.split(":", 1)[1].strip()
    return ""


def first(value: Any) -> Any:
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


def admissible(info: Mapping[str, Any]) -> List[str]:
    value = info.get("admissible_commands", [])
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [str(command) for command in value]


def trim_history(history: List[str], max_history: int) -> List[str]:
    if len(history) <= max_history:
        return history
    if max_history < 3:
        raise ValueError("max_history must be >=3")
    return history[:1] + history[-(max_history - 1) :]


def replace_action_in_generation(generation: str, action: str) -> str:
    match = re.search(r"Action:\s*", generation, flags=re.IGNORECASE)
    thought = generation[: match.start()].rstrip() if match else generation.rstrip()
    return "{}{}Action: {}".format(thought, "\n" if thought else "", action)


class Runtime(object):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        print("[counterfactual] loading model {}".format(args.model_dir), flush=True)
        self.model = SDARForCausalLM.from_pretrained(
            args.model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        self.model_identity = build_model_identity(args.model_dir, SDARForCausalLM)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_dir, trust_remote_code=True
        )
        mask_ids = self.tokenizer("<|MASK|>", add_special_tokens=False)["input_ids"]
        self.mask_id = int(mask_ids[0] if isinstance(mask_ids, list) else mask_ids)
        generation_config = GenerationConfig.from_pretrained(args.model_dir)
        eos = generation_config.eos_token_id
        self.stop_ids = list(eos) if isinstance(eos, list) else [int(eos)]
        with open(args.prompt_json, "r", encoding="utf-8") as handle:
            self.prompts = json.load(handle)

    def decision(
        self,
        goal: str,
        history: Sequence[str],
        commands: Sequence[str],
        decision_seed: int,
        include_trace: bool,
    ) -> Dict[str, Any]:
        prompt_text = build_prompt(self.tokenizer, self.prompts, goal, list(history))
        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=False
        )["input_ids"]
        seed_everything(int(decision_seed))
        started = time.time()
        decision = two_stage_decision_decode(
            self.model,
            self.tokenizer,
            prompt_ids,
            list(commands),
            self.mask_id,
            self.stop_ids,
            thought_order=self.args.thought_order,
            action_order=self.args.action_order,
            action_grammar=self.args.action_grammar,
            gen_length=self.args.gen_length,
            action_gen_length=self.args.action_gen_length,
            block_length=self.args.block_length,
            denoising_steps=self.args.denoising_steps,
            temperature=self.args.temperature,
        )
        raw_action = str(decision["raw_action"])
        if self.args.action_grammar == "trie" and raw_action not in commands:
            raise ValueError("Trie Action is not admissible: {!r}".format(raw_action))
        result: Dict[str, Any] = {
            "prompt": prompt_text,
            "prompt_sha256": sha256_json(prompt_ids),
            "generation": decision["response_text"],
            "raw_action": raw_action,
            "executed_action": raw_action,
            "decision_seed": int(decision_seed),
            "decision_latency_seconds": time.time() - started,
        }
        if include_trace:
            result["prompt_ids"] = [int(value) for value in prompt_ids]
            result["response_ids"] = [int(value) for value in decision["response_ids"]]
            result["decision_traces"] = decision["decision_traces"]
        return result

    def policy_action_sample(
        self,
        action_prefix_ids: Sequence[int],
        commands: Sequence[str],
        decision_seed: int,
    ) -> str:
        """Resample only Action from the old policy, holding base Thought fixed."""

        if self.args.action_order != "ar" or self.args.action_grammar != "trie":
            raise ValueError("policy Action intervention requires canonical AR+Trie Action")
        trie = build_action_trie(self.tokenizer, list(commands))

        def allowed(committed):
            return trie.allowed_next(committed)

        prefix = [int(value) for value in action_prefix_ids]
        tensor = torch.tensor([prefix], dtype=torch.long, device=self.model.device)
        seed_everything(int(decision_seed))
        output, raw_trace = exact_ar_action_generate(
            self.model,
            tensor,
            self.mask_id,
            gen_length=self.args.action_gen_length,
            temperature=self.args.temperature,
            stop_ids=self.stop_ids,
            constraint=allowed,
            constraint_name="trie",
        )
        trace = trim_generation_trace(
            output[len(prefix) :].tolist(),
            raw_trace["step_map"].tolist(),
            raw_trace["behavior_logprobs"].tolist(),
            self.mask_id,
            self.stop_ids,
            confidence=raw_trace["commit_confidence"].tolist(),
            sampling=raw_trace["sampling"],
        )
        action = self.tokenizer.decode(
            trace["response_ids"], skip_special_tokens=True
        ).strip()
        if action.endswith("."):
            action = action[:-1].strip()
        if action not in commands:
            raise ValueError("old-policy Action intervention is not admissible: {!r}".format(action))
        return action


def environment_manager(args: argparse.Namespace):
    import yaml
    import alfworld.agents.environment as envs

    with open(args.alfworld_config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    manifest = load_split_manifest(args.split_manifest)
    split_spec = manifest["splits"][args.split_name]
    split_root = args.split_root or split_spec["source_root_hint"]
    split_key = (
        "eval_id_data_path" if args.split_name == "dev_seen" else "eval_ood_data_path"
    )
    config["dataset"][split_key] = split_root
    env_split = (
        "eval_in_distribution"
        if args.split_name == "dev_seen"
        else "eval_out_of_distribution"
    )
    EnvClass = envs.get_environment(config["env"]["type"])
    manager = EnvClass(config, train_eval=env_split)
    frozen_ids = [row["game_id"] for row in scan_games(split_root)]
    check_game_ids(frozen_ids, split_spec)
    ordered_files = order_game_files_by_manifest(manager.game_files, frozen_ids)
    return manager, dict(zip(frozen_ids, ordered_files)), split_spec, split_root


def make_repeated_env(manager: Any, game_file: str, resets: int) -> Any:
    if resets < 1:
        raise ValueError("environment requires at least one reset")
    manager.game_files = [game_file] * int(resets)
    if hasattr(manager, "num_games"):
        manager.num_games = len(manager.game_files)
    return manager.init_env(batch_size=1)


def close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def collect_base_episode(
    runtime: Runtime,
    env: Any,
    game_id: str,
    rollout_id: int,
    base_rollout_seed: int,
    max_steps: int,
    max_history: int,
) -> Dict[str, Any]:
    reset_seed = stable_seed(base_rollout_seed, game_id, rollout_id, "env-reset")
    seed_everything(reset_seed)
    observations, info = env.reset()
    observation = str(first(observations))
    commands = admissible(info)
    initial_state = make_state_record(observation, commands, 0.0, False)
    goal = extract_goal(observation)
    history = ["Observation: {}".format(canonical_observation(observation))]
    episode_id = "cf-episode-{}".format(
        sha256_json(
            [
                game_id,
                int(rollout_id),
                int(base_rollout_seed),
                runtime.model_identity["identity_sha256"],
            ]
        )[:20]
    )
    turns: List[Dict[str, Any]] = []
    score = 0.0
    done = False
    previous_score = 0.0
    current_state = initial_state
    for turn_id in range(max_steps):
        if done:
            break
        decision_seed = stable_seed(
            base_rollout_seed, game_id, rollout_id, turn_id, "decision"
        )
        decision = runtime.decision(
            goal, history, commands, decision_seed, include_trace=True
        )
        action = decision["executed_action"]
        observations, rewards, dones, info = env.step([action])
        observation = str(first(observations))
        score = float(first(rewards))
        done = bool(first(dones))
        next_commands = admissible(info)
        next_state = make_state_record(observation, next_commands, score, done)
        turns.append(
            {
                "turn_id": turn_id,
                "state_before": current_state,
                "prompt": decision["prompt"],
                "prompt_sha256": decision["prompt_sha256"],
                "generation": decision["generation"],
                "prompt_ids": decision["prompt_ids"],
                "response_ids": decision["response_ids"],
                "decision_traces": decision["decision_traces"],
                "raw_action": decision["raw_action"],
                "executed_action": action,
                "decision_seed": decision_seed,
                "decision_latency_seconds": decision["decision_latency_seconds"],
                "turn_reward": score - previous_score,
                "state_after": next_state,
            }
        )
        previous_score = score
        current_state = next_state
        commands = next_commands
        history.append(history_action(action))
        history.append("Observation: {}".format(canonical_observation(observation)))
        history = trim_history(history, max_history)
    if len(turns) < 3:
        raise ValueError(
            "base episode {} ended after {} turns; cannot form distinct O01 anchors".format(
                episode_id, len(turns)
            )
        )
    return {
        "game_id": game_id,
        "episode_id": episode_id,
        "rollout_id": int(rollout_id),
        "base_rollout_seed": int(base_rollout_seed),
        "env_reset_seed": int(reset_seed),
        "goal": goal,
        "initial_state": initial_state,
        "horizon": len(turns),
        "return": score,
        "success": bool(score >= 1.0 or done),
        "turns": turns,
    }


def run_base(args: argparse.Namespace, runtime: Runtime) -> None:
    manager, game_files, split_spec, split_root = environment_manager(args)
    game_ids = list(game_files.keys())[args.game_offset : args.game_offset + args.num_games]
    if len(game_ids) != args.num_games:
        raise ValueError("requested base game slice is outside the frozen split")
    artifact: Dict[str, Any] = {
        "contract_version": COUNTERFACTUAL_CONTRACT_VERSION,
        "artifact_kind": BASE_ARTIFACT_KIND,
        "complete": False,
        "memory_contract": MEMORY_CONTRACT,
        "model_identity_sha256": runtime.model_identity["identity_sha256"],
        "model_identity": runtime.model_identity,
        "seed_registration": args.seed_registration,
        "split": {
            "name": args.split_name,
            "manifest": os.path.realpath(args.split_manifest),
            "game_ids_sha256": split_spec["game_ids_sha256"],
            "source_root": os.path.realpath(split_root),
            "game_offset": args.game_offset,
            "game_ids": game_ids,
        },
        "protocol": {
            "base_rollout_seed": args.base_rollout_seed,
            "rollouts_per_game": args.rollouts_per_game,
            "max_steps": args.max_steps,
            "max_history": args.max_history,
            "thought_order": args.thought_order,
            "action_order": args.action_order,
            "action_grammar": args.action_grammar,
            "temperature": args.temperature,
            "agent_memory_note": (
                "Only executed Action and resulting Observation enter future prompts; "
                "Thought-only effect is structural zero under this artifact."
            ),
        },
        "base_episodes": [],
        "anchors": [],
    }
    atomic_json_dump(artifact, args.out)
    for game_index, game_id in enumerate(game_ids):
        env = make_repeated_env(manager, game_files[game_id], args.rollouts_per_game)
        try:
            for rollout_id in range(args.rollouts_per_game):
                episode = collect_base_episode(
                    runtime,
                    env,
                    game_id,
                    rollout_id,
                    args.base_rollout_seed,
                    args.max_steps,
                    args.max_history,
                )
                artifact["base_episodes"].append(episode)
                artifact["anchors"] = build_anchor_records(artifact["base_episodes"])
                atomic_json_dump(artifact, args.out)
                print(
                    "[counterfactual/base] game={}/{} rollout={}/{} H={} return={:.3f}".format(
                        game_index + 1,
                        len(game_ids),
                        rollout_id + 1,
                        args.rollouts_per_game,
                        episode["horizon"],
                        episode["return"],
                    ),
                    flush=True,
                )
        finally:
            close_env(env)
    artifact["complete"] = True
    artifact["n_base_episodes"] = len(artifact["base_episodes"])
    artifact["n_anchors"] = len(artifact["anchors"])
    atomic_json_dump(artifact, args.out)
    print(
        "[counterfactual/base] PASS episodes={} anchors={} out={}".format(
            artifact["n_base_episodes"], artifact["n_anchors"], args.out
        ),
        flush=True,
    )


def load_base_artifact(path: str, runtime: Runtime) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("contract_version") != COUNTERFACTUAL_CONTRACT_VERSION:
        raise ValueError("unsupported base counterfactual contract")
    if artifact.get("artifact_kind") != BASE_ARTIFACT_KIND or not artifact.get("complete"):
        raise ValueError("base counterfactual artifact is incomplete")
    if artifact.get("memory_contract") != MEMORY_CONTRACT:
        raise ValueError("base artifact memory contract drift")
    if not model_identities_match(artifact["model_identity"], runtime.model_identity):
        raise ValueError("base artifact and branch runner use different behavior checkpoints")
    registered = artifact.get("seed_registration", {})
    if registered.get("sha256") != runtime.args.seed_registration.get("sha256"):
        raise ValueError("base and branch stages use different frozen seed manifests")
    if registered.get("values") != runtime.args.seed_registration.get("values"):
        raise ValueError("base and branch stages use different registered seeds")
    anchors = build_anchor_records(artifact["base_episodes"])
    if anchors != artifact.get("anchors"):
        raise ValueError("base artifact anchor index is inconsistent")
    return artifact


def branch_environment_step(
    env: Any,
    action: str,
) -> Tuple[str, float, bool, Mapping[str, Any], Dict[str, Any]]:
    observations, rewards, dones, info = env.step([action])
    observation = str(first(observations))
    score = float(first(rewards))
    done = bool(first(dones))
    state = make_state_record(observation, admissible(info), score, done)
    return observation, score, done, info, state


def run_one_branch(
    runtime: Runtime,
    env: Any,
    episode: Mapping[str, Any],
    anchor: Mapping[str, Any],
    schedule: Mapping[str, Any],
    branch_role: str,
    intervention: Mapping[str, Any],
    max_steps: int,
    max_history: int,
) -> Dict[str, Any]:
    prefix = replay_environment_prefix(
        env,
        episode,
        anchor["turn_id"],
        schedule["env_reset_seed"],
        seed_everything,
        max_history=max_history,
    )
    history = list(prefix["history"])
    info = prefix["info"]
    commands = admissible(info)
    score = float(prefix["cumulative_reward"])
    done = bool(prefix["done"])
    base_turn = episode["turns"][int(anchor["turn_id"])]
    if branch_role == "original":
        anchor_action = str(base_turn["executed_action"])
        anchor_generation = str(base_turn["generation"])
        anchor_decision_seed = int(base_turn["decision_seed"])
    elif branch_role == "alternative":
        kind = str(intervention["kind"])
        if kind == "uniform_admissible_action":
            anchor_action = str(intervention["action"])
            anchor_generation = replace_action_in_generation(
                str(base_turn["generation"]), anchor_action
            )
            anchor_decision_seed = None
        elif kind == "old_policy_action_sample":
            anchor_decision_seed = int(intervention["policy_decision_seed"])
            action_trace = base_turn.get("decision_traces", {}).get("action")
            if not action_trace:
                raise ValueError("base turn lacks the fixed-Thought Action prefix")
            anchor_action = runtime.policy_action_sample(
                action_trace["prefix_ids"],
                commands,
                anchor_decision_seed,
            )
            anchor_generation = replace_action_in_generation(
                str(base_turn["generation"]), anchor_action
            )
        elif kind in ("thought_resampled_action", "explicit_counterfactual_action"):
            # V4-O02/O03 supplies an Action regenerated from a controlled
            # Thought step/token or canonical Action-token replacement.  The
            # Action must still pass the same canonical AR+Trie contract below;
            # no parser/snap mapping is introduced here.
            anchor_action = str(intervention["action"])
            anchor_generation = str(intervention["generation"])
            anchor_decision_seed = int(intervention["policy_decision_seed"])
        else:
            raise ValueError("unsupported intervention kind: {}".format(kind))
    else:
        raise ValueError("branch_role must be original or alternative")
    if runtime.args.action_grammar == "trie" and anchor_action not in commands:
        raise ValueError("branch anchor Action is not admissible")
    observation, score, done, info, state = branch_environment_step(env, anchor_action)
    if branch_role == "original":
        expected = base_turn["state_after"]
        if state["state_sha256"] != expected["state_sha256"]:
            raise ValueError("original anchor transition did not reproduce base episode")
    trajectory: List[Dict[str, Any]] = [
        {
            "turn_offset": 0,
            "global_turn_id": int(anchor["turn_id"]),
            "decision_seed": anchor_decision_seed,
            "generation": anchor_generation,
            "raw_action": anchor_action,
            "executed_action": anchor_action,
            "state_after_sha256": state["state_sha256"],
            "reward": score,
            "done": done,
        }
    ]
    history.append(history_action(anchor_action))
    history.append("Observation: {}".format(canonical_observation(observation)))
    history = trim_history(history, max_history)
    for future_offset, decision_seed in enumerate(
        schedule["continuation_decision_seeds"]
    ):
        global_turn_id = int(anchor["turn_id"]) + 1 + future_offset
        if done or global_turn_id >= max_steps:
            break
        commands = admissible(info)
        decision = runtime.decision(
            episode["goal"],
            history,
            commands,
            int(decision_seed),
            include_trace=False,
        )
        action = decision["executed_action"]
        observation, score, done, info, state = branch_environment_step(env, action)
        trajectory.append(
            {
                "turn_offset": future_offset + 1,
                "global_turn_id": global_turn_id,
                "decision_seed": int(decision_seed),
                "generation": decision["generation"],
                "raw_action": decision["raw_action"],
                "executed_action": action,
                "state_after_sha256": state["state_sha256"],
                "reward": score,
                "done": done,
            }
        )
        history.append(history_action(action))
        history.append("Observation: {}".format(canonical_observation(observation)))
        history = trim_history(history, max_history)
    result = {
        "branch_role": branch_role,
        "anchor_action": anchor_action,
        "return": score,
        "success": bool(score >= 1.0 or done),
        "final_state_sha256": state["state_sha256"],
        "crn_schedule_sha256": schedule["schedule_sha256"],
        "prefix_replay": prefix["audit"],
        "original_anchor_transition_reproduced": (
            True if branch_role == "original" else None
        ),
        "trajectory": trajectory,
    }
    result["reproducibility_fingerprint"] = branch_reproducibility_fingerprint(result)
    return result


def make_intervention(
    args: argparse.Namespace,
    anchor: Mapping[str, Any],
    episode: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> Dict[str, Any]:
    turn = episode["turns"][int(anchor["turn_id"])]
    original_action = str(turn["executed_action"])
    commands = list(turn["state_before"]["admissible_commands"])
    seed = int(schedule["intervention_seed"])
    use_policy = (
        args.intervention_policy in ("policy_action", "mixture")
        and (
            args.intervention_policy == "policy_action"
            or random.Random(seed).random() < args.policy_mixture_probability
        )
    )
    if use_policy:
        return {
            "kind": "old_policy_action_sample",
            "distribution": "old_policy_action_conditional_on_fixed_base_thought",
            "policy_decision_seed": stable_seed(seed, "policy-action"),
            "memory_semantics": MEMORY_CONTRACT,
        }
    return {
        "kind": "uniform_admissible_action",
        "distribution": "uniform_admissible_excluding_original",
        "action": choose_uniform_alternative_action(commands, original_action, seed),
        "memory_semantics": MEMORY_CONTRACT,
    }


def local_game_slice(args: argparse.Namespace, base: Mapping[str, Any]) -> List[str]:
    game_ids = list(base["split"]["game_ids"])
    count = len(game_ids) if args.branch_num_games < 0 else args.branch_num_games
    selected = game_ids[args.branch_game_offset : args.branch_game_offset + count]
    if len(selected) != count:
        raise ValueError("requested branch game slice is outside the base artifact")
    return selected


def run_branches(args: argparse.Namespace, runtime: Runtime) -> None:
    base = load_base_artifact(args.base_artifact, runtime)
    manager, discovered, split_spec, _ = environment_manager(args)
    if split_spec["game_ids_sha256"] != base["split"]["game_ids_sha256"]:
        raise ValueError("runtime split differs from base artifact split")
    selected_games = local_game_slice(args, base)
    all_anchor_ids = [row["anchor_id"] for row in base["anchors"]]
    global_retest = set(
        select_retest_anchor_ids(
            all_anchor_ids, args.retest_fraction, args.selection_seed
        )
    )
    local_anchor_meta = [
        row for row in base["anchors"] if row["game_id"] in set(selected_games)
    ]
    local_retest = sorted(
        row["anchor_id"] for row in local_anchor_meta if row["anchor_id"] in global_retest
    )
    episode_by_id = {
        row["episode_id"]: row for row in base["base_episodes"]
    }
    repro_anchor_ids = set(
        row["anchor_id"]
        for row in sorted(local_anchor_meta, key=lambda value: value["anchor_id"])[
            : args.repro_anchor_count
        ]
    )
    artifact: Dict[str, Any] = {
        "contract_version": COUNTERFACTUAL_CONTRACT_VERSION,
        "artifact_kind": BRANCH_ARTIFACT_KIND,
        "phase": args.phase,
        "complete": False,
        "memory_contract": MEMORY_CONTRACT,
        "model_identity_sha256": runtime.model_identity["identity_sha256"],
        "seed_registration": args.seed_registration,
        "base_artifact_path": os.path.realpath(args.base_artifact),
        "base_artifact_sha256": sha256_file(args.base_artifact),
        "protocol": {
            "k": args.k,
            "branch_seed": args.branch_seed,
            "selection_seed": args.selection_seed,
            "retest_fraction": args.retest_fraction,
            "retest_anchor_ids": local_retest,
            "global_retest_anchor_ids_sha256": sha256_json(sorted(global_retest)),
            "intervention_policy": args.intervention_policy,
            "policy_mixture_probability": args.policy_mixture_probability,
            "max_steps": base["protocol"]["max_steps"],
            "max_history": base["protocol"]["max_history"],
            "crn_note": (
                "Original and alternative share env-reset and every post-anchor "
                "decision seed; branch label is absent from seed derivation."
            ),
            "thought_only_reference": "structural-zero-not-run-under-current-memory-contract",
        },
        "game_ids": selected_games,
        "anchors": [],
        "reproducibility_checks": [],
    }
    atomic_json_dump(artifact, args.out)
    anchors_by_game: Dict[str, List[Mapping[str, Any]]] = {}
    for anchor in local_anchor_meta:
        anchors_by_game.setdefault(anchor["game_id"], []).append(anchor)
    max_steps = int(base["protocol"]["max_steps"])
    max_history = int(base["protocol"]["max_history"])
    for game_index, game_id in enumerate(selected_games):
        game_anchors = sorted(
            anchors_by_game.get(game_id, []), key=lambda row: row["anchor_id"]
        )
        resets = 0
        for anchor in game_anchors:
            repeats = 2 if anchor["anchor_id"] in global_retest else 1
            resets += repeats * args.k * 2
            if anchor["anchor_id"] in repro_anchor_ids:
                resets += 2
        env = make_repeated_env(manager, discovered[game_id], resets)
        try:
            for anchor in game_anchors:
                episode = episode_by_id[anchor["episode_id"]]
                anchor_result: Dict[str, Any] = {
                    **dict(anchor),
                    "original_action": episode["turns"][int(anchor["turn_id"])][
                        "executed_action"
                    ],
                    "repeats": {},
                }
                repeat_ids = [0] + ([1] if anchor["anchor_id"] in global_retest else [])
                first_pair = None
                for repeat_id in repeat_ids:
                    pairs = []
                    for sample_id in range(args.k):
                        future_turns = max_steps - int(anchor["turn_id"]) - 1
                        schedule = make_crn_schedule(
                            args.branch_seed,
                            anchor["anchor_id"],
                            repeat_id,
                            sample_id,
                            future_turns,
                        )
                        intervention = make_intervention(
                            args, anchor, episode, schedule
                        )
                        original = run_one_branch(
                            runtime,
                            env,
                            episode,
                            anchor,
                            schedule,
                            "original",
                            intervention,
                            max_steps,
                            max_history,
                        )
                        alternative = run_one_branch(
                            runtime,
                            env,
                            episode,
                            anchor,
                            schedule,
                            "alternative",
                            intervention,
                            max_steps,
                            max_history,
                        )
                        pair = {
                            "sample_id": sample_id,
                            "crn_schedule": schedule,
                            "intervention": intervention,
                            "original": original,
                            "alternative": alternative,
                            "paired_effect": original["return"] - alternative["return"],
                        }
                        if first_pair is None:
                            first_pair = pair
                        pairs.append(pair)
                    anchor_result["repeats"][str(repeat_id)] = {
                        "pairs": pairs,
                        "reference": summarize_paired_effects(pairs),
                    }
                if anchor["anchor_id"] in repro_anchor_ids:
                    if first_pair is None:
                        raise ValueError("reproducibility anchor has no branch pair")
                    for role in ("original", "alternative"):
                        duplicate = run_one_branch(
                            runtime,
                            env,
                            episode,
                            anchor,
                            first_pair["crn_schedule"],
                            role,
                            first_pair["intervention"],
                            max_steps,
                            max_history,
                        )
                        expected = first_pair[role]["reproducibility_fingerprint"]
                        observed = duplicate["reproducibility_fingerprint"]
                        artifact["reproducibility_checks"].append(
                            {
                                "check_id": "{}-{}".format(anchor["anchor_id"], role),
                                "anchor_id": anchor["anchor_id"],
                                "branch_role": role,
                                "schedule_sha256": first_pair["crn_schedule"][
                                    "schedule_sha256"
                                ],
                                "expected_fingerprint": expected,
                                "duplicate_fingerprint": observed,
                                "pass": expected == observed,
                            }
                        )
                artifact["anchors"].append(anchor_result)
                atomic_json_dump(artifact, args.out)
                effect = anchor_result["repeats"]["0"]["reference"]["mean_effect"]
                se = anchor_result["repeats"]["0"]["reference"]["standard_error"]
                print(
                    "[counterfactual/branch] game={}/{} anchor={} {} h={} effect={:.3f} se={}".format(
                        game_index + 1,
                        len(selected_games),
                        anchor["anchor_id"],
                        anchor["stratum"],
                        anchor["turn_id"],
                        effect,
                        "NA" if se is None else "{:.3f}".format(se),
                    ),
                    flush=True,
                )
        finally:
            close_env(env)
    artifact["complete"] = True
    artifact["n_anchors"] = len(artifact["anchors"])
    artifact["n_reproducibility_checks"] = len(artifact["reproducibility_checks"])
    atomic_json_dump(artifact, args.out)
    print(
        "[counterfactual/branch] DONE anchors={} checks={} out={}".format(
            artifact["n_anchors"], artifact["n_reproducibility_checks"], args.out
        ),
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--mode", choices=("base", "branch"), required=True)
    result.add_argument("--model_dir", required=True)
    result.add_argument("--alfworld_config", required=True)
    result.add_argument("--prompt_json", required=True)
    result.add_argument("--split_manifest", required=True)
    result.add_argument("--counterfactual_seed_manifest", required=True)
    result.add_argument("--seed_phase", choices=("screening", "submission"), default="screening")
    result.add_argument("--seed_replicate", type=int, default=0)
    result.add_argument("--split_name", choices=("dev_seen", "final_unseen"), default="dev_seen")
    result.add_argument("--split_root")
    result.add_argument("--out", required=True)
    result.add_argument("--thought_order", choices=("ao", "ar"), default="ao")
    result.add_argument("--action_order", choices=("ao", "ar"), default="ar")
    result.add_argument("--action_grammar", choices=("none", "trie"), default="trie")
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--gen_length", type=int, default=128)
    result.add_argument("--action_gen_length", type=int, default=24)
    result.add_argument("--block_length", type=int, default=4)
    result.add_argument("--denoising_steps", type=int, default=4)
    result.add_argument("--max_history", type=int, default=24)
    result.add_argument("--max_steps", type=int, default=30)
    result.add_argument("--num_games", type=int, default=16)
    result.add_argument("--game_offset", type=int, default=0)
    result.add_argument("--rollouts_per_game", type=int, default=2)
    result.add_argument("--base_rollout_seed", type=int, default=62001)
    result.add_argument("--base_artifact")
    result.add_argument("--branch_game_offset", type=int, default=0)
    result.add_argument("--branch_num_games", type=int, default=-1)
    result.add_argument("--k", type=int, default=4)
    result.add_argument("--branch_seed", type=int, default=72001)
    result.add_argument("--selection_seed", type=int, default=82001)
    result.add_argument("--retest_fraction", type=float, default=0.25)
    result.add_argument(
        "--intervention_policy",
        choices=("policy_action", "uniform_action", "mixture"),
        default="policy_action",
    )
    result.add_argument("--policy_mixture_probability", type=float, default=0.5)
    result.add_argument("--repro_anchor_count", type=int, default=1)
    result.add_argument("--phase", choices=("e04_smoke", "o01"), default="e04_smoke")
    return result


def validate_args(args: argparse.Namespace) -> None:
    if args.action_grammar == "trie" and args.action_order != "ar":
        raise ValueError("canonical Trie Action requires action_order=ar")
    if args.num_games < 1 or args.rollouts_per_game < 1 or args.max_steps < 3:
        raise ValueError("base protocol requires games>=1, rollouts>=1, max_steps>=3")
    if args.k < 1 or args.repro_anchor_count < 1:
        raise ValueError("branch protocol requires K>=1 and at least one duplicate check")
    if not (0.0 <= args.retest_fraction <= 1.0):
        raise ValueError("retest_fraction must be in [0,1]")
    if not (0.0 <= args.policy_mixture_probability <= 1.0):
        raise ValueError("policy_mixture_probability must be in [0,1]")
    if args.mode == "branch" and not args.base_artifact:
        raise ValueError("branch mode requires --base_artifact")


def main() -> None:
    args = parser().parse_args()
    validate_args(args)
    args.seed_registration = validate_registered_seeds(args)
    runtime = Runtime(args)
    if args.mode == "base":
        run_base(args, runtime)
    else:
        run_branches(args, runtime)


if __name__ == "__main__":
    main()
