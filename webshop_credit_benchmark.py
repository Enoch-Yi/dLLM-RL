#!/usr/bin/env python3
"""Submission-scale WebShop counterfactual credit benchmark (NeDA V4-2).

The protocol mirrors the ALFWorld benchmark but uses a frozen WebShop task
split, a per-job local server, the canonical finite Action interface in
``neda_webshop_env``, and exact replay through unique fixed-task sessions.
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Mapping, Sequence

from alfworld_counterfactual import (
    Runtime,
    atomic_json_dump,
    replace_action_in_generation,
    trim_history,
    validate_registered_seeds,
)
from alfworld_credit_benchmark import sample_policy_action_excluding
from alfworld_step_token_counterfactual import (
    action_from_modified_thought,
    resample_recorded_action_coordinate,
    resample_recorded_thought_coordinate,
)
from neda_counterfactual import make_crn_schedule, summarize_paired_effects
from neda_credit_benchmark import quantile_anchor_turns
from neda_data_contract import history_action
from neda_repro import (
    model_identities_match,
    seed_everything,
    sha256_file,
    sha256_json,
    stable_seed,
)
from neda_step_token_counterfactual import (
    REPLACEMENT_CONTRACT,
    replacement_positions,
    select_action_token_coordinates,
    select_step_token_coordinates,
)
from neda_webshop_env import (
    ACTION_INTERFACE,
    MEMORY_CONTRACT,
    WebShopSession,
    replay_prefix,
)
from neda_webshop_credit_contract import (
    BASE_CONTRACT_VERSION,
    BASE_GATE_CONTRACT_VERSION,
    BRANCH_CONTRACT_VERSION,
    build_webshop_anchors,
)
from neda_joint_policy import JOINT_METHODS, load_dcolt_head
from neda_v4_decision import two_stage_decision_decode
from r002_alfworld import build_action_trie


def _prompt(runtime: Runtime, goal: str, history: Sequence[str], commands: Sequence[str]) -> str:
    prompts = runtime.prompts
    examples = prompts.get("examples", "")
    if isinstance(examples, list):
        examples = "\n".join(str(value) for value in examples)
    command_text = "\n".join("- {}".format(value) for value in commands)
    user = (
        "{}\n\n{}\n\nHere is the task:\n{}\n\n{}\n\n"
        "The only admissible Actions for this turn are:\n{}\n"
        "Now give the next step in the format 'Thought: ...\\nAction: ...'."
    ).format(
        prompts.get("instruction", ""),
        examples,
        goal,
        "\n".join(history),
        command_text,
    )
    messages = [
        {"role": "system", "content": prompts.get("system_msg", "You are a helpful assistant.")},
        {"role": "user", "content": user},
    ]
    return runtime.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


class WebRuntime(Runtime):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.rl_method = getattr(args, "rl_method", None)
        self.position_head = None
        self.dcolt_head_metadata = None
        if self.rl_method is not None:
            self.rl_method = str(self.rl_method).lower()
            if self.rl_method not in JOINT_METHODS:
                raise ValueError("unknown joint WebShop RL method")
        head_path = getattr(args, "dcolt_head_path", None)
        if self.rl_method == "dcolt":
            if not head_path or not os.path.isfile(head_path):
                raise ValueError("DCoLT WebShop rollout requires --dcolt_head_path")
            self.position_head, self.dcolt_head_metadata = load_dcolt_head(
                self.model.config, head_path, map_location="cpu"
            )
            self.position_head = self.position_head.to(
                device=self.model.device,
                dtype=next(self.model.parameters()).dtype,
            ).eval()
        elif head_path:
            raise ValueError("--dcolt_head_path is only valid for DCoLT")

    def decision(
        self,
        goal: str,
        history: Sequence[str],
        commands: Sequence[str],
        decision_seed: int,
        include_trace: bool,
    ) -> Dict[str, Any]:
        prompt_text = _prompt(self, goal, history, commands)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
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
            rl_method=self.rl_method,
            position_temperature=float(
                getattr(self.args, "position_temperature", 0.5)
            ),
            position_head=self.position_head,
            neda_credit_boundaries=int(
                getattr(self.args, "neda_credit_boundaries", 4)
            ),
        )
        action = str(decision["raw_action"])
        if action not in commands:
            raise ValueError("WebShop Trie Action is outside the canonical Action set")
        result: Dict[str, Any] = {
            "prompt": prompt_text,
            "prompt_sha256": sha256_json(prompt_ids),
            "generation": decision["response_text"],
            "raw_action": action,
            "executed_action": action,
            "decision_seed": int(decision_seed),
            "decision_latency_seconds": time.time() - started,
        }
        if include_trace:
            result["prompt_ids"] = [int(value) for value in prompt_ids]
            result["response_ids"] = [int(value) for value in decision["response_ids"]]
            result["decision_traces"] = decision["decision_traces"]
        return result


def _load_split(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("contract_version") != "neda-webshop-v4-splits-v1":
        raise ValueError("unsupported frozen WebShop split")
    split = value.get("splits", {}).get("credit_dev", {})
    ids = [int(item) for item in split.get("task_ids", [])]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("WebShop credit_dev split must contain 100 unique task IDs")
    if split.get("task_ids_sha256") != sha256_json(ids):
        raise ValueError("WebShop credit_dev task ID SHA drift")
    return {"manifest": value, "split": split, "task_ids": ids}


def _asset_identity(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("contract_version") != "neda-webshop-assets-v1":
        raise ValueError("unsupported frozen WebShop asset manifest")
    return {
        "path": os.path.realpath(path),
        "file_sha256": sha256_file(path),
        "canonical_sha256": sha256_json(value),
        "index_file_set_sha256": str(value["index"]["file_set_sha256"]),
        "expected_documents": int(value["index"]["expected_documents"]),
        "server_seed": int(value["runtime"]["webshop_server_seed"]),
    }


def _session_factory(args: argparse.Namespace, task_id: int, stem: str):
    def build(namespace: str):
        return WebShopSession(
            args.agentboard_root,
            task_id,
            "{}-{}".format(stem, namespace),
            web_url=args.web_url,
        )

    return build


def collect_base_episode(
    runtime: WebRuntime,
    args: argparse.Namespace,
    task_id: int,
    task_index: int,
    rollout_id: int,
    validate_credit_anchor_coverage: bool = True,
) -> Dict[str, Any]:
    namespace = "web-base-t{}-r{}-{}".format(task_index, rollout_id, os.getpid())
    session = _session_factory(args, task_id, namespace)("episode")
    reset_seed = stable_seed(args.base_rollout_seed, task_id, rollout_id, "web-reset")
    seed_everything(reset_seed)
    current = session.reset()
    goal = current["goal"]
    history = ["Observation: {}".format(current["observation"])]
    initial_state = current["state"]
    episode_id = "web-credit-episode-{}".format(
        sha256_json(
            [
                int(task_id),
                int(rollout_id),
                int(args.base_rollout_seed),
                runtime.model_identity["identity_sha256"],
                ACTION_INTERFACE,
            ]
        )[:20]
    )
    turns: List[Dict[str, Any]] = []
    previous_progress = float(current["progress"])
    for turn_id in range(int(args.max_steps)):
        if current["done"]:
            break
        commands = list(current["actions"])
        decision_seed = stable_seed(
            args.base_rollout_seed, task_id, rollout_id, turn_id, "web-decision"
        )
        decision = runtime.decision(goal, history, commands, decision_seed, True)
        before = current["state"]
        action_contract = current["action_contract"]
        current = session.step(decision["executed_action"])
        turns.append(
            {
                "turn_id": turn_id,
                "state_before": before,
                "action_contract": action_contract,
                "prompt": decision["prompt"],
                "prompt_sha256": decision["prompt_sha256"],
                "generation": decision["generation"],
                "prompt_ids": decision["prompt_ids"],
                "response_ids": decision["response_ids"],
                "decision_traces": decision["decision_traces"],
                "raw_action": decision["raw_action"],
                "executed_action": decision["executed_action"],
                "backend_action": current["backend_action"],
                "decision_seed": int(decision_seed),
                "decision_latency_seconds": decision["decision_latency_seconds"],
                "turn_reward": float(current["progress"]) - previous_progress,
                "terminal_reward": float(current["reward"]),
                "grounding": bool(current["grounding"]),
                "state_after": current["state"],
            }
        )
        previous_progress = float(current["progress"])
        history.append(history_action(decision["executed_action"]))
        history.append("Observation: {}".format(current["observation"]))
        history = trim_history(history, int(args.max_history))
    if validate_credit_anchor_coverage:
        quantile_anchor_turns(len(turns), int(args.anchors_per_episode))
    return {
        "game_id": "webshop-fixed-{}".format(task_id),
        "webshop_task_id": int(task_id),
        "episode_id": episode_id,
        "rollout_id": int(rollout_id),
        "base_rollout_seed": int(args.base_rollout_seed),
        "env_reset_seed": int(reset_seed),
        "goal": goal,
        "initial_state": initial_state,
        "horizon": len(turns),
        "return": float(current["progress"]),
        "terminal_reward": float(current["reward"]),
        "success": bool(float(current["reward"]) == 1.0 and current["done"]),
        "turns": turns,
    }


def _base_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "task_index": int(args.task_index),
        "rollouts": int(args.rollouts),
        "anchors_per_episode": int(args.anchors_per_episode),
        "max_steps": int(args.max_steps),
        "max_history": int(args.max_history),
        "thought_order": str(args.thought_order),
        "action_order": str(args.action_order),
        "action_grammar": str(args.action_grammar),
        "action_interface": ACTION_INTERFACE,
        "temperature": float(args.temperature),
        "base_rollout_seed": int(args.base_rollout_seed),
        "code_bundle_sha256": str(args.code_bundle_sha256),
    }


def run_base(args: argparse.Namespace, runtime: WebRuntime) -> None:
    split = _load_split(args.split_manifest)
    task_id = split["task_ids"][int(args.task_index)]
    assets = _asset_identity(args.asset_manifest)
    protocol = _base_protocol(args)
    if os.path.isfile(args.out):
        artifact = json.load(open(args.out, "r", encoding="utf-8"))
        if (
            artifact.get("contract_version") != BASE_CONTRACT_VERSION
            or int(artifact.get("webshop_task_id", -1)) != task_id
            or artifact.get("protocol") != protocol
            or not model_identities_match(artifact.get("model_identity", {}), runtime.model_identity)
        ):
            raise ValueError("existing WebShop credit base checkpoint drift")
        if artifact.get("complete"):
            print("[credit-web/base] already complete: {}".format(args.out), flush=True)
            return
    else:
        artifact = {
            "contract_version": BASE_CONTRACT_VERSION,
            "artifact_kind": "stateful-credit-base",
            "environment": "webshop",
            "complete": False,
            "memory_contract": MEMORY_CONTRACT,
            "action_interface": ACTION_INTERFACE,
            "code_bundle_sha256": str(args.code_bundle_sha256),
            "model_identity": runtime.model_identity,
            "model_identity_sha256": runtime.model_identity["identity_sha256"],
            "seed_registration": args.seed_registration,
            "split": {
                "name": "credit_dev",
                "manifest": os.path.realpath(args.split_manifest),
                "manifest_sha256": sha256_file(args.split_manifest),
                "task_ids_sha256": split["split"]["task_ids_sha256"],
            },
            "assets": assets,
            "task_index": int(args.task_index),
            "webshop_task_id": int(task_id),
            "game_id": "webshop-fixed-{}".format(task_id),
            "protocol": protocol,
            "base_episodes": [],
            "anchors": [],
        }
        atomic_json_dump(artifact, args.out)
    completed = {int(row["rollout_id"]) for row in artifact["base_episodes"]}
    for rollout_id in range(int(args.rollouts)):
        if rollout_id in completed:
            continue
        episode = collect_base_episode(
            runtime, args, task_id, int(args.task_index), rollout_id
        )
        artifact["base_episodes"].append(episode)
        artifact["anchors"] = build_webshop_anchors(
            artifact["base_episodes"], int(args.anchors_per_episode)
        )
        atomic_json_dump(artifact, args.out)
        print(
            "[credit-web/base] task={} rollout={}/{} H={} progress={:.3f} success={}".format(
                args.task_index,
                rollout_id + 1,
                args.rollouts,
                episode["horizon"],
                episode["return"],
                episode["success"],
            ),
            flush=True,
        )
    expected = int(args.rollouts) * int(args.anchors_per_episode)
    if len(artifact["base_episodes"]) != int(args.rollouts) or len(artifact["anchors"]) != expected:
        raise ValueError("WebShop credit base completion count drift")
    artifact["complete"] = True
    artifact["n_episodes"] = len(artifact["base_episodes"])
    artifact["n_anchors"] = len(artifact["anchors"])
    artifact["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json_dump(artifact, args.out)
    gate = {
        "contract_version": BASE_GATE_CONTRACT_VERSION,
        "status": "PASS",
        "environment": "webshop",
        "task_index": int(args.task_index),
        "webshop_task_id": int(task_id),
        "base_artifact": os.path.realpath(args.out),
        "base_artifact_sha256": sha256_file(args.out),
        "model_identity_sha256": runtime.model_identity["identity_sha256"],
        "code_bundle_sha256": str(args.code_bundle_sha256),
        "asset_manifest_sha256": assets["file_sha256"],
        "n_episodes": len(artifact["base_episodes"]),
        "n_anchors": len(artifact["anchors"]),
    }
    gate["gate_sha256"] = sha256_json(gate)
    atomic_json_dump(gate, args.gate)
    print("[credit-web/base] PASS out={}".format(args.out), flush=True)


def load_base(args: argparse.Namespace, runtime: WebRuntime) -> Dict[str, Any]:
    artifact = json.load(open(args.base, "r", encoding="utf-8"))
    if (
        artifact.get("contract_version") != BASE_CONTRACT_VERSION
        or artifact.get("artifact_kind") != "stateful-credit-base"
        or not artifact.get("complete")
        or artifact.get("memory_contract") != MEMORY_CONTRACT
        or artifact.get("action_interface") != ACTION_INTERFACE
    ):
        raise ValueError("WebShop credit base is incomplete or incompatible")
    if not model_identities_match(artifact["model_identity"], runtime.model_identity):
        raise ValueError("WebShop credit base and branch checkpoint differ")
    if artifact.get("seed_registration") != args.seed_registration:
        raise ValueError("WebShop credit base/branch seed registration drift")
    if artifact.get("code_bundle_sha256") != str(args.code_bundle_sha256):
        raise ValueError("WebShop credit base/branch code bundle drift")
    if sha256_file(args.base) != str(args.expected_base_sha256):
        raise ValueError("WebShop credit base SHA differs from its gate")
    return artifact


def _summary_pairs(
    pairs: Sequence[Mapping[str, Any]], originals: Mapping[str, Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    return [
        {
            "crn_schedule": row["crn_schedule"],
            "original": originals[str(row["sample_id"])],
            "alternative": row["alternative"],
            "paired_effect": float(row["paired_effect"]),
        }
        for row in pairs
    ]


def _find_reference(rows: Sequence[Mapping[str, Any]], coordinate_id: str):
    for row in rows:
        if str(row.get("coordinate_id")) == str(coordinate_id):
            return row
    return None


def run_one_branch(
    runtime: WebRuntime,
    args: argparse.Namespace,
    episode: Mapping[str, Any],
    anchor: Mapping[str, Any],
    schedule: Mapping[str, Any],
    branch_role: str,
    intervention: Mapping[str, Any],
    namespace: str,
) -> Dict[str, Any]:
    factory = _session_factory(args, int(episode["webshop_task_id"]), namespace)
    seed_everything(int(schedule["env_reset_seed"]))
    prefix = replay_prefix(
        factory,
        episode,
        int(anchor["turn_id"]),
        "prefix",
        int(args.max_history),
        env_reset_seed=int(schedule["env_reset_seed"]),
    )
    session = prefix["session"]
    current = prefix["current"]
    history = list(prefix["history"])
    base_turn = episode["turns"][int(anchor["turn_id"])]
    if branch_role == "original":
        action = str(base_turn["executed_action"])
        generation = str(base_turn["generation"])
        decision_seed = int(base_turn["decision_seed"])
    elif branch_role == "alternative":
        action = str(intervention["action"])
        generation = str(intervention["generation"])
        decision_seed = int(intervention["policy_decision_seed"])
    else:
        raise ValueError("WebShop branch role must be original or alternative")
    if action not in current["actions"]:
        raise ValueError("WebShop branch anchor Action is not canonical")
    current = session.step(action)
    reproduced = current["state"]["state_sha256"] == base_turn["state_after"]["state_sha256"]
    if branch_role == "original" and not reproduced:
        raise ValueError("WebShop original anchor transition did not reproduce")
    trajectory: List[Dict[str, Any]] = [
        {
            "turn_offset": 0,
            "global_turn_id": int(anchor["turn_id"]),
            "decision_seed": decision_seed,
            "generation": generation,
            "raw_action": action,
            "executed_action": action,
            "backend_action": current["backend_action"],
            "state_after_sha256": current["state"]["state_sha256"],
            "progress": float(current["progress"]),
            "terminal_reward": float(current["reward"]),
            "done": bool(current["done"]),
        }
    ]
    history.extend(
        [history_action(action), "Observation: {}".format(current["observation"])]
    )
    history = trim_history(history, int(args.max_history))
    for future_offset, future_seed in enumerate(schedule["continuation_decision_seeds"]):
        global_turn = int(anchor["turn_id"]) + 1 + future_offset
        if current["done"] or global_turn >= int(args.max_steps):
            break
        decision = runtime.decision(
            episode["goal"], history, current["actions"], int(future_seed), False
        )
        current = session.step(decision["executed_action"])
        trajectory.append(
            {
                "turn_offset": future_offset + 1,
                "global_turn_id": global_turn,
                "decision_seed": int(future_seed),
                "generation": decision["generation"],
                "raw_action": decision["raw_action"],
                "executed_action": decision["executed_action"],
                "backend_action": current["backend_action"],
                "state_after_sha256": current["state"]["state_sha256"],
                "progress": float(current["progress"]),
                "terminal_reward": float(current["reward"]),
                "done": bool(current["done"]),
            }
        )
        history.extend(
            [
                history_action(decision["executed_action"]),
                "Observation: {}".format(current["observation"]),
            ]
        )
        history = trim_history(history, int(args.max_history))
    return {
        "branch_role": branch_role,
        "crn_schedule_sha256": schedule["schedule_sha256"],
        "prefix_replay": prefix["audit"],
        "anchor_action": action,
        "original_anchor_transition_reproduced": bool(reproduced),
        "trajectory": trajectory,
        "return": float(current["progress"]),
        "terminal_reward": float(current["reward"]),
        "success": bool(float(current["reward"]) == 1.0 and current["done"]),
        "horizon_from_anchor": len(trajectory),
    }


def run_branch(args: argparse.Namespace, runtime: WebRuntime) -> None:
    base = load_base(args, runtime)
    split = _load_split(args.split_manifest)
    if base["split"]["task_ids_sha256"] != split["split"]["task_ids_sha256"]:
        raise ValueError("WebShop credit branch split drift")
    if not (0 <= int(args.anchor_index) < len(base["anchors"])):
        raise ValueError("WebShop credit anchor index is outside the fixed base")
    anchor = base["anchors"][int(args.anchor_index)]
    episode = next(
        row for row in base["base_episodes"] if row["episode_id"] == anchor["episode_id"]
    )
    turn = episode["turns"][int(anchor["turn_id"])]
    thought_trace = turn["decision_traces"]["thought"]
    action_trace = turn["decision_traces"]["action"]
    commands = list(turn["state_before"]["admissible_actions"])
    thought_selection = select_step_token_coordinates(
        thought_trace,
        anchor["anchor_id"],
        args.selection_seed,
        block_size=args.block_length,
        n_steps=2,
        tokens_per_step=2,
    )
    trie = build_action_trie(runtime.tokenizer, commands)
    if not trie.is_complete(action_trace["response_ids"]):
        raise ValueError("base WebShop Action trace is outside its frozen Trie")
    action_selection = select_action_token_coordinates(
        action_trace,
        anchor["anchor_id"],
        args.selection_seed,
        trie.allowed_next,
        max_action_tokens=2,
        min_action_tokens=1,
    )
    selection = {"thought": thought_selection, "action": action_selection}
    selection["selection_sha256"] = sha256_json(selection)
    coordinates = replacement_positions(thought_selection) + list(
        action_selection["coordinates"]
    )
    protocol = {
        "k": int(args.k),
        "branch_seed": int(args.branch_seed),
        "selection_seed": int(args.selection_seed),
        "n_thought_steps": 2,
        "thought_tokens_per_step": 2,
        "min_action_tokens": 1,
        "max_action_tokens": 2,
        "replacement_contract": REPLACEMENT_CONTRACT,
        "turn_intervention": "old-policy-Action-conditional-hybrid-exact-v2",
        "turn_intervention_max_attempts": 128,
        "local_intervention": REPLACEMENT_CONTRACT,
        "max_steps": int(base["protocol"]["max_steps"]),
        "max_history": int(base["protocol"]["max_history"]),
        "logprob_tolerance": float(args.logprob_tolerance),
        "code_bundle_sha256": str(args.code_bundle_sha256),
    }
    if os.path.isfile(args.out):
        artifact = json.load(open(args.out, "r", encoding="utf-8"))
        if (
            artifact.get("contract_version") != BRANCH_CONTRACT_VERSION
            or artifact.get("base_artifact_sha256") != str(args.expected_base_sha256)
            or artifact.get("anchor") != anchor
            or artifact.get("protocol") != protocol
            or artifact.get("selection") != selection
        ):
            raise ValueError("existing WebShop credit branch checkpoint drift")
        if artifact.get("complete"):
            print("[credit-web/branch] already complete: {}".format(args.out), flush=True)
            return
    else:
        artifact = {
            "contract_version": BRANCH_CONTRACT_VERSION,
            "artifact_kind": "counterfactual-credit-reference",
            "environment": "webshop",
            "complete": False,
            "memory_contract": MEMORY_CONTRACT,
            "action_interface": ACTION_INTERFACE,
            "code_bundle_sha256": str(args.code_bundle_sha256),
            "model_identity_sha256": runtime.model_identity["identity_sha256"],
            "seed_registration": args.seed_registration,
            "base_artifact": os.path.realpath(args.base),
            "base_artifact_sha256": str(args.expected_base_sha256),
            "task_index": int(args.task_index),
            "webshop_task_id": int(episode["webshop_task_id"]),
            "anchor_index": int(args.anchor_index),
            "anchor": anchor,
            "protocol": protocol,
            "selection": selection,
            "originals": {},
            "turn_pairs": [],
            "local_references": [],
        }
        atomic_json_dump(artifact, args.out)
    schedules = {}
    for sample_id in range(int(args.k)):
        schedule = make_crn_schedule(
            args.branch_seed,
            anchor["anchor_id"],
            0,
            sample_id,
            max(0, int(protocol["max_steps"]) - int(anchor["turn_id"]) - 1),
        )
        schedules[sample_id] = schedule
        key = str(sample_id)
        if key not in artifact["originals"]:
            artifact["originals"][key] = run_one_branch(
                runtime,
                args,
                episode,
                anchor,
                schedule,
                "original",
                {},
                "orig-a{}-k{}".format(args.anchor_index, sample_id),
            )
            atomic_json_dump(artifact, args.out)
    existing_turn = {int(row["sample_id"]) for row in artifact["turn_pairs"]}
    for sample_id in range(int(args.k)):
        if sample_id in existing_turn:
            continue
        schedule = schedules[sample_id]
        action_seed = stable_seed(
            schedule["intervention_seed"], "turn-action-excluding-original"
        )
        conditional = sample_policy_action_excluding(
            runtime,
            action_trace["prefix_ids"],
            commands,
            turn["executed_action"],
            action_seed,
            max_attempts=protocol["turn_intervention_max_attempts"],
        )
        alternative_action = conditional["accepted_action"]
        regenerated = {
            "action": alternative_action,
            "generation": replace_action_in_generation(turn["generation"], alternative_action),
            "policy_decision_seed": conditional["accepted_decision_seed"],
            "conditional_sampling": conditional,
        }
        alternative = run_one_branch(
            runtime,
            args,
            episode,
            anchor,
            schedule,
            "alternative",
            regenerated,
            "turn-a{}-k{}".format(args.anchor_index, sample_id),
        )
        original = artifact["originals"][str(sample_id)]
        artifact["turn_pairs"].append(
            {
                "sample_id": sample_id,
                "crn_schedule": schedule,
                "original_key": str(sample_id),
                "alternative_action": regenerated,
                "alternative": alternative,
                "paired_effect": float(original["return"]) - float(alternative["return"]),
            }
        )
        atomic_json_dump(artifact, args.out)
    for coordinate in coordinates:
        if _find_reference(artifact["local_references"], coordinate["coordinate_id"]):
            continue
        pairs = []
        for sample_id in range(int(args.k)):
            schedule = schedules[sample_id]
            replacement_seed = stable_seed(
                schedule["intervention_seed"],
                coordinate["coordinate_id"],
                "{}-replacement".format(coordinate["level"]),
            )
            if coordinate["level"] == "action_token":
                action_result = resample_recorded_action_coordinate(
                    runtime,
                    action_trace,
                    coordinate,
                    commands,
                    replacement_seed,
                    args.logprob_tolerance,
                    turn["generation"],
                )
                replacement = action_result["replacement"]
                regenerated = action_result["regenerated_action"]
            else:
                replacement = resample_recorded_thought_coordinate(
                    runtime,
                    thought_trace,
                    coordinate,
                    replacement_seed,
                    args.block_length,
                    args.gen_length,
                    args.logprob_tolerance,
                )
                action_seed = stable_seed(
                    schedule["intervention_seed"],
                    coordinate["coordinate_id"],
                    "regenerated-action",
                )
                regenerated = action_from_modified_thought(
                    runtime,
                    thought_trace,
                    replacement["modified_response_ids"],
                    commands,
                    action_seed,
                )
            alternative = run_one_branch(
                runtime,
                args,
                episode,
                anchor,
                schedule,
                "alternative",
                regenerated,
                "local-a{}-{}-k{}".format(
                    args.anchor_index, coordinate["coordinate_id"][-8:], sample_id
                ),
            )
            original = artifact["originals"][str(sample_id)]
            pairs.append(
                {
                    "sample_id": sample_id,
                    "crn_schedule": schedule,
                    "original_key": str(sample_id),
                    "replacement": replacement,
                    "regenerated_action": regenerated,
                    "action_changed": regenerated["action"] != turn["executed_action"],
                    "alternative": alternative,
                    "paired_effect": float(original["return"]) - float(alternative["return"]),
                }
            )
        artifact["local_references"].append(
            {
                **coordinate,
                "pairs": pairs,
                "reference": summarize_paired_effects(_summary_pairs(pairs, artifact["originals"])),
            }
        )
        atomic_json_dump(artifact, args.out)
        print(
            "[credit-web/branch] task={} anchor={} level={} refs={}/{}".format(
                args.task_index,
                args.anchor_index,
                coordinate["level"],
                len(artifact["local_references"]),
                len(coordinates),
            ),
            flush=True,
        )
    if set(artifact["originals"]) != {str(index) for index in range(int(args.k))}:
        raise ValueError("WebShop shared original completion drift")
    if len(artifact["turn_pairs"]) != int(args.k):
        raise ValueError("WebShop turn reference completion drift")
    if len(artifact["local_references"]) != len(coordinates):
        raise ValueError("WebShop local reference completion drift")
    artifact["turn_reference"] = summarize_paired_effects(
        _summary_pairs(artifact["turn_pairs"], artifact["originals"])
    )
    artifact["complete"] = True
    artifact["n_local_references"] = len(artifact["local_references"])
    artifact["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json_dump(artifact, args.out)
    print("[credit-web/branch] PASS out={}".format(args.out), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--mode", choices=("base", "branch"), required=True)
    result.add_argument("--model_dir", required=True)
    result.add_argument("--prompt_json", required=True)
    result.add_argument("--split_manifest", required=True)
    result.add_argument("--asset_manifest", required=True)
    result.add_argument("--counterfactual_seed_manifest", required=True)
    result.add_argument("--agentboard_root", required=True)
    result.add_argument("--web_url", default="http://127.0.0.1:3000")
    result.add_argument("--seed_phase", default="submission")
    result.add_argument("--seed_replicate", type=int, default=0)
    result.add_argument("--task_index", type=int, required=True)
    result.add_argument("--rollouts", type=int, default=2)
    result.add_argument("--anchors_per_episode", type=int, default=3)
    result.add_argument("--anchor_index", type=int, default=0)
    result.add_argument("--base")
    result.add_argument("--expected_base_sha256", default="")
    result.add_argument("--out", required=True)
    result.add_argument("--gate")
    result.add_argument("--thought_order", default="ao")
    result.add_argument("--action_order", default="ar")
    result.add_argument("--action_grammar", default="trie")
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--gen_length", type=int, default=128)
    result.add_argument("--action_gen_length", type=int, default=64)
    result.add_argument("--block_length", type=int, default=4)
    result.add_argument("--denoising_steps", type=int, default=4)
    result.add_argument("--max_history", type=int, default=24)
    result.add_argument("--max_steps", type=int, default=20)
    result.add_argument("--base_rollout_seed", type=int, default=62001)
    result.add_argument("--branch_seed", type=int, default=72001)
    result.add_argument("--selection_seed", type=int, default=82001)
    result.add_argument("--k", type=int, default=4)
    result.add_argument("--logprob_tolerance", type=float, default=0.05)
    result.add_argument("--code_bundle_sha256", default="")
    return result


def main() -> None:
    args = parser().parse_args()
    if (
        args.thought_order != "ao"
        or args.action_order != "ar"
        or args.action_grammar != "trie"
        or args.block_length != 4
        or args.denoising_steps != 4
    ):
        raise ValueError("WebShop credit requires canonical AO-Thought/AR+Trie-Action")
    if not (0 <= int(args.task_index) < 100):
        raise ValueError("WebShop credit task index must be in 0..99")
    if args.mode == "base" and not args.gate:
        raise ValueError("WebShop base mode requires --gate")
    if args.mode == "branch" and (not args.base or not args.expected_base_sha256):
        raise ValueError("WebShop branch mode requires frozen base path and SHA")
    if len(str(args.code_bundle_sha256)) != 64:
        raise ValueError("WebShop credit requires a frozen 64-hex code bundle SHA")
    args.seed_registration = validate_registered_seeds(args)
    runtime = WebRuntime(args)
    if args.mode == "base":
        run_base(args, runtime)
    else:
        run_branch(args, runtime)


if __name__ == "__main__":
    main()
