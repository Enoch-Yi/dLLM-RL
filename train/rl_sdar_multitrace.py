#!/usr/bin/env python3
"""One-optimizer recorded-path learner for NeDA AO Thought + AR Action.

Unlike the legacy homogeneous rl_sdar batches, each replay row keeps its native
prefix and replay width.  A batch size of one permits AO Thought and fixed-width
AR Action rows to coexist without changing either learner coordinate.  All old
scores are frozen from the behavior checkpoint before the first optimizer step.
"""

import glob
import importlib
import inspect
import json
import logging
import math
import os
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, InitProcessGroupKwargs, set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# NGC 24.10 alpha-torch compatibility shims shared with rl_sdar.py.
try:
    import torch.distributed.tensor as _dt
    if not hasattr(_dt, "DTensor"):
        from torch.distributed._tensor import DTensor as _DTensor
        _dt.DTensor = _DTensor
except Exception:
    pass
try:
    import transformers.modeling_utils as _mu
    if getattr(_mu, "DTensor", None) is None:
        from torch.distributed._tensor import DTensor as _DTensorMu
        _mu.DTensor = _DTensorMu
except Exception:
    pass

from models import SDARForCausalLM
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_error, set_verbosity_info
from neda_repro import sha256_file, sha256_json
from neda_data_contract import NATIVE_THOUGHT_SCORING_LAYOUT
from neda_joint_policy import (
    JointSDARPolicy,
    load_dcolt_head,
    mapg_position_scores,
    plackett_luce_logprob,
    save_dcolt_head,
)
from neda_torch_replay import (
    exact_replay_numerics,
    make_absolute_block_duplicate_attention,
    make_basic_block_attention,
)
from neda_v4_multitrace import (
    MULTITRACE_CONTRACT_VERSION,
    NEDA_TOKEN_ABLATION_METHODS,
    build_native_rows,
    distributed_step_plan,
    pad_native_rows_for_distributed,
)
from train.utils import flatten_omega_conf, get_config


logger = get_logger(__name__, log_level="INFO")


REPLAY_DIAGNOSTIC_CONTRACT_VERSION = "neda-v4-replay-drift-diagnostic-v1"
POSITION_RATIO_METHODS = ("mapg", "dcolt", "neda")
WRAPPED_POLICY_METHODS = POSITION_RATIO_METHODS + NEDA_TOKEN_ABLATION_METHODS


def _drift_coordinates(
    batch: Mapping[str, Any], score: torch.Tensor, process_index: int
) -> List[Dict[str, Any]]:
    """Return authenticated-coordinate metadata for one native replay row.

    The full token tensors stay in the already SHA-bound learner artifacts.
    This diagnostic records stable hashes plus the exact selected coordinate,
    which is enough to recover an offender without copying long prompts into
    every training receipt.
    """

    prediction_mask = batch["prediction_mask"]
    if prediction_mask.ndim != 2 or prediction_mask.shape[0] != 1:
        raise ValueError("replay drift diagnostic requires a size-one row")
    positions = prediction_mask[0].nonzero(as_tuple=True)[0].tolist()
    start_pos = int(batch["start_pos"])
    width = int(batch["response_width"])
    extended = [
        int(value)
        for value in batch["extended_input_ids"][0].detach().cpu().tolist()
    ]
    position_ids = [
        int(value) for value in batch["position_ids"][0].detach().cpu().tolist()
    ]
    prefix = extended[:start_pos]
    learner_state = extended[start_pos : start_pos + width]
    row_identity = {
        "sample_id": str(batch["sample_id"]),
        "source": str(batch["source"]),
        "round_id": int(batch["round_id"]),
        "start_pos": start_pos,
        "response_width": width,
        "block_size": int(batch["block_size"]),
        "attention_layout": str(batch["attention_layout"]),
        "extended_input_ids_sha256": sha256_json(extended),
        "prefix_ids_sha256": sha256_json(prefix),
        "learner_state_sha256": sha256_json(learner_state),
        "position_ids_sha256": sha256_json(position_ids),
    }
    result = []
    for position in positions:
        stored = float(batch["rollout_logp"][0, position].detach().float().cpu())
        learner = float(score[0, position].detach().float().cpu())
        result.append(
            {
                **row_identity,
                "process_index": int(process_index),
                "dataset_index": int(batch["index"]),
                "selected_position": int(position),
                "response_position": int(position) - start_pos,
                "label_token_id": int(
                    batch["labels"][0, position].detach().cpu()
                ),
                "stored_behavior_logprob": stored,
                "learner_logprob": learner,
                "signed_error": learner - stored,
                "abs_error": abs(learner - stored),
            }
        )
    return result


def _top_drift_coordinates(
    per_rank: Sequence[Sequence[Mapping[str, Any]]], limit: int
) -> List[Dict[str, Any]]:
    """Merge rank-local coordinate diagnostics deterministically."""

    if int(limit) <= 0:
        raise ValueError("replay diagnostic top-k must be positive")
    unique: Dict[Any, Dict[str, Any]] = {}
    for values in per_rank:
        for raw in values or []:
            value = dict(raw)
            key = (
                int(value["dataset_index"]),
                int(value["selected_position"]),
                str(value["source"]),
            )
            previous = unique.get(key)
            if previous is None or float(value["abs_error"]) > float(
                previous["abs_error"]
            ):
                unique[key] = value
    ordered = sorted(
        unique.values(),
        key=lambda value: (
            -float(value["abs_error"]),
            str(value["source"]),
            str(value["sample_id"]),
            int(value["round_id"]),
            int(value["selected_position"]),
            int(value["dataset_index"]),
        ),
    )
    return ordered[: int(limit)]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_checkpoint(model, tokenizer, config, accelerator, name: str) -> None:
    """Save a complete portable checkpoint without importing the legacy trainer."""

    output_dir = Path(config.experiment.project)
    save_base = output_dir / "ckpt"
    save_base.mkdir(parents=True, exist_ok=True)
    model_to_save = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        save_dir = save_base / str(name)
        joint = isinstance(model_to_save, JointSDARPolicy)
        policy = model_to_save.policy if joint else model_to_save
        if joint:
            policy_state = {
                key[len("policy.") :]: value
                for key, value in state_dict.items()
                if key.startswith("policy.")
            }
        else:
            policy_state = state_dict
        policy.save_pretrained(
            save_dir,
            save_function=accelerator.save,
            state_dict=policy_state,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(save_dir))
        if joint and model_to_save.position_head is not None:
            head_state = {
                key[len("position_head.") :]: value
                for key, value in state_dict.items()
                if key.startswith("position_head.")
            }
            save_dcolt_head(
                model_to_save.position_head,
                str(save_dir / "dcolt_upm.pt"),
                seed=int(config.training.get("position_head_seed", 0)),
                source_model=str(config.model.pretrained_model),
                extra={
                    "trained_with": str(config.training.registered_method),
                    "full_parameter_policy_update": True,
                },
                state_dict=head_state,
            )
        modules = set()
        for value in (policy, getattr(policy, "config", None), tokenizer):
            if value is not None and getattr(value.__class__, "__module__", None):
                modules.add(value.__class__.__module__)
        copied = 0
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                source = inspect.getsourcefile(module)
                if not source or not os.path.isfile(source):
                    continue
                source_dir = os.path.dirname(source)
                for pattern in (
                    "modeling_*.py", "configuration_*.py",
                    "tokenization_*.py", "processing_*.py",
                ):
                    for filename in glob.glob(os.path.join(source_dir, pattern)):
                        destination = save_dir / os.path.basename(filename)
                        if not destination.exists():
                            shutil.copy2(filename, destination)
                            copied += 1
            except Exception as error:
                logger.warning("Skip copying from module %s: %s", module_name, error)
        with (save_base / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump({"save_time": time.strftime("%Y-%m-%d %H:%M:%S")}, handle, indent=2)
            handle.write("\n")
        logger.info("Saved model + tokenizer to %s (copied %d dynamic files)", save_dir, copied)


class NativeReplayDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, Any]]):
        self.rows = [dict(row) for row in rows]
        self.old_logp: List[torch.Tensor] = [
            torch.full((len(row["labels"]),), float("-inf"), dtype=torch.float32)
            for row in self.rows
        ]
        self.row_entropy: List[float] = [float("nan")] * len(self.rows)
        self.old_position_logp: List[float] = [float("nan")] * len(self.rows)
        self.active: List[bool] = [
            not bool(row.get("is_distributed_padding", False))
            for row in self.rows
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        return {
            "index": int(index),
            "sample_id": row["sample_id"],
            "source": row["source"],
            "round_id": int(row["round_id"]),
            "start_pos": int(row["start_pos"]),
            "response_width": int(row["response_width"]),
            "block_size": int(row["block_size"]),
            "attention_layout": str(row["attention_layout"]),
            "extended_input_ids": torch.as_tensor(row["extended_input_ids"], dtype=torch.long),
            "prediction_mask": torch.as_tensor(row["prediction_mask"], dtype=torch.bool),
            "labels": torch.as_tensor(row["labels"], dtype=torch.long),
            "position_ids": torch.as_tensor(row["position_ids"], dtype=torch.long),
            "adv_map": torch.as_tensor(row["adv_map"], dtype=torch.float32),
            "rollout_logp": torch.as_tensor(row["rollout_logp"], dtype=torch.float32),
            "constraint_allowed_token_ids": row["constraint_allowed_token_ids"],
            "registered_method": row["registered_method"],
            "credit_contract": row["credit_contract"],
            "step_selection": row.get("step_selection"),
            "position_decision": row.get("position_decision"),
            "position_mask_index": (
                None
                if row.get("position_mask_index") is None
                else torch.as_tensor(row["position_mask_index"], dtype=torch.bool)
            ),
            "step_credit": (
                None
                if row.get("step_credit") is None
                else torch.as_tensor(float(row["step_credit"]), dtype=torch.float32)
            ),
            "is_distributed_padding": bool(
                row.get("is_distributed_padding", False)
            ),
            "active": bool(self.active[index]),
        }


def single_collate(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("native multitrace learner requires batch_size_lm=1")
    row = batch[0]
    result = dict(row)
    for key in (
        "extended_input_ids", "prediction_mask", "labels", "position_ids",
        "adv_map", "rollout_logp",
    ):
        result[key] = row[key].unsqueeze(0)
    if row.get("position_mask_index") is not None:
        result["position_mask_index"] = row["position_mask_index"].unsqueeze(0)
    return result


def _projected_logp(
    model,
    batch: Mapping[str, Any],
    return_entropy: bool = False,
    return_position: bool = False,
) -> Any:
    start_pos = int(batch["start_pos"])
    width = int(batch["response_width"])
    block_size = int(batch["block_size"])
    extended = batch["extended_input_ids"]
    if str(batch.get("attention_layout")) == NATIVE_THOUGHT_SCORING_LAYOUT:
        attention = make_absolute_block_duplicate_attention(
            start_pos + 2 * width, start_pos, block_size, device=extended.device
        )
    else:
        attention = make_basic_block_attention(
            start_pos + 2 * width, start_pos, block_size, device=extended.device
        )
    position_decision = batch.get("position_decision")
    registered_method = str(batch.get("registered_method", ""))
    is_dcolt = registered_method == "dcolt"
    position_kwargs = {}
    if position_decision is not None and str(
        position_decision.get("policy")
    ) == "dcolt_upm":
        current_block = torch.zeros(
            (1, width), dtype=torch.bool, device=extended.device
        )
        current_block[
            0,
            torch.as_tensor(
                position_decision["current_block_positions"],
                dtype=torch.long,
                device=extended.device,
            ),
        ] = True
        position_kwargs = {
            "position_response_start": start_pos,
            "position_response_width": width,
            "position_timestep": torch.as_tensor(
                [float(position_decision["timestep"])],
                dtype=torch.float32,
                device=extended.device,
            ),
            "position_mask_index": batch["position_mask_index"],
            "position_current_block": current_block,
        }
    elif is_dcolt:
        # ZeRO-3 requires every rank to traverse the same parameter graph in
        # the same order.  Native replay deliberately interleaves Thought
        # rows (which have a learned DCoLT position decision) and Action rows
        # (which do not).  Before this dummy branch, a distributed microstep
        # could therefore run the 252.8M-parameter UPM on only some ranks;
        # job 18455153 then observed an UPM ALLGATHER on rank 0 while the
        # other ranks had already reached a BROADCAST, and timed out.
        #
        # Action credit remains token-only.  These shape-valid inputs merely
        # execute the UPM collectively; the zero-valued graph anchor below
        # makes its mathematical contribution exactly zero while retaining
        # identical forward/backward parameter hooks on every rank.
        dummy_mask = torch.zeros(
            (1, width), dtype=torch.bool, device=extended.device
        )
        position_kwargs = {
            "position_response_start": start_pos,
            "position_response_width": width,
            "position_timestep": torch.zeros(
                1, dtype=torch.float32, device=extended.device
            ),
            "position_mask_index": dummy_mask,
            "position_current_block": dummy_mask.clone(),
        }
    outputs = model(
        input_ids=extended,
        attention_mask=attention,
        position_ids=batch["position_ids"],
        use_cache=False,
        store_kv=False,
        **position_kwargs,
    )
    logits = outputs.logits
    projected = torch.cat(
        [logits[:, :start_pos, :], logits[:, start_pos + width :, :]], dim=1
    )
    prediction_mask = batch["prediction_mask"]
    selected_logits = projected[prediction_mask].float()
    selected_labels = batch["labels"][prediction_mask]
    selected_log_probs = F.log_softmax(selected_logits, dim=-1)
    selected_scores = selected_log_probs.gather(
        -1, selected_labels.unsqueeze(-1)
    ).squeeze(-1)
    scores = torch.zeros_like(batch["labels"], dtype=torch.float32)
    scores[prediction_mask] = selected_scores
    if is_dcolt:
        if outputs.position_logits is None:
            raise RuntimeError(
                "DCoLT collective-safe replay requires an UPM forward on "
                "every native row"
            )
        # Preserve the UPM autograd hooks even for Action rows and for forced
        # final Thought commitments whose Plackett--Luce term is exactly zero.
        # This scalar is identically zero and cannot change token log-probs or
        # the registered DCoLT objective.
        scores = scores + outputs.position_logits.float().sum() * 0.0
    entropy_values = -(
        selected_log_probs.exp() * selected_log_probs
    ).sum(dim=-1)
    selected_positions = [
        int(position)
        for position in prediction_mask[0].nonzero(as_tuple=True)[0].tolist()
    ]
    entropy_by_position = {
        position: entropy_values[index]
        for index, position in enumerate(selected_positions)
    }
    allowed_rows = batch.get("constraint_allowed_token_ids", [])
    if allowed_rows:
        scores = scores.clone()
        for response_position, allowed in enumerate(allowed_rows):
            if allowed is None:
                continue
            allowed = [int(value) for value in allowed]
            position = start_pos + response_position
            target = int(batch["labels"][0, position])
            if target not in allowed:
                raise ValueError("learner target is outside recorded trie support")
            if not bool(prediction_mask[0, position]):
                continue
            logits = projected[0, position, :].float()
            allowed_tensor = torch.as_tensor(
                allowed, dtype=torch.long, device=logits.device
            )
            allowed_logits = logits.index_select(0, allowed_tensor)
            allowed_log_probs = F.log_softmax(allowed_logits, dim=0)
            target_index = allowed.index(target)
            scores[0, position] = allowed_log_probs[target_index]
            entropy_by_position[position] = -(
                allowed_log_probs.exp() * allowed_log_probs
            ).sum()
    position_logprob = None
    if position_decision is not None:
        policy = str(position_decision["policy"])
        if policy == "mapg_logit":
            position_scores = mapg_position_scores(
                projected[:, start_pos : start_pos + width, :]
            )[0]
        elif policy == "dcolt_upm":
            position_scores = outputs.position_logits[0]
        else:
            raise ValueError("unknown recorded position policy")
        position_logprob = plackett_luce_logprob(
            position_scores,
            position_decision["candidate_positions"],
            position_decision["selected_positions"],
            float(position_decision["temperature"]),
        )
    if not return_entropy:
        if return_position:
            return scores, position_logprob
        return scores
    if not entropy_by_position:
        raise ValueError("native replay row has no selected-token entropy")
    entropy = torch.stack(list(entropy_by_position.values())).mean()
    if return_position:
        return scores, entropy, position_logprob
    return scores, entropy


def main() -> None:
    config = get_config()
    if int(config.training.batch_size_lm) != 1:
        raise ValueError("multitrace learner requires batch_size_lm=1")
    if str(config.training.method) != "exact_multitrace":
        raise ValueError("multitrace learner requires training.method=exact_multitrace")
    seed = int(config.training.get("train_seed", config.training.seed))
    registered_method = str(config.training.get("registered_method", "neda"))
    set_seed(seed)
    if bool(config.training.enable_tf32):
        raise ValueError("recorded-path multitrace learner requires TF32 disabled")

    config.experiment.logging_dir = str(Path(config.experiment.project) / "logs")
    wandb_mode = os.environ.get("WANDB_MODE", "online").strip().lower()
    wandb_tracking_requested = wandb_mode != "disabled"
    accelerator_kwargs = {}
    process_group_timeout_seconds = None
    if registered_method == "dcolt":
        process_group_timeout_seconds = int(
            os.environ.get("NEDA_DCOLT_PROCESS_GROUP_TIMEOUT_SECONDS", "3600")
        )
        if not 600 <= process_group_timeout_seconds <= 7200:
            raise ValueError("DCoLT process-group timeout must be in [600, 7200]")
        # This handler must be passed by the learner at the first Accelerator
        # construction.  Wrapping the entry point after Accelerate/DeepSpeed
        # has initialized its state leaves the default 600-second watchdog in
        # place, which is exactly what jobs 18454571/18454572 exposed.
        accelerator_kwargs["kwargs_handlers"] = [
            InitProcessGroupKwargs(
                timeout=timedelta(seconds=process_group_timeout_seconds)
            )
        ]
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config.training.gradient_accumulation_steps),
        mixed_precision=str(config.training.mixed_precision),
        log_with="wandb" if wandb_tracking_requested else None,
        project_dir=config.experiment.logging_dir,
        # Each native replay row has its own prefix/width and the collator is
        # deliberately batch-size one.  On multi-GPU jobs each process must
        # therefore receive a different complete row; a size-one batch cannot
        # be split across processes.
        split_batches=False,
        **accelerator_kwargs,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()
    tracker_ready = False
    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        if wandb_tracking_requested:
            try:
                accelerator.init_trackers(
                    os.environ.get(
                        "WANDB_PROJECT", str(config.experiment.project)
                    ),
                    config=dict(flatten_omega_conf(config, resolve=True)),
                    init_kwargs={
                        "wandb": {"name": str(config.model.optimized_name)}
                    },
                )
                tracker_ready = True
            except Exception as error:
                logger.warning(
                    "wandb init failed; JSONL-only training continues: %s",
                    error,
                )

    pretrained_model = str(config.model.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    base_model = SDARForCausalLM.from_pretrained(
        pretrained_model, trust_remote_code=True, torch_dtype="auto"
    )
    position_head_metadata = None
    position_head = None
    if registered_method in WRAPPED_POLICY_METHODS:
        if registered_method == "dcolt":
            position_head_path = os.path.realpath(
                str(config.training.get("position_head_path", ""))
            )
            if not os.path.isfile(position_head_path):
                raise ValueError("DCoLT learner requires position_head_path")
            position_head, position_head_metadata = load_dcolt_head(
                base_model.config, position_head_path, map_location="cpu"
            )
            position_head = position_head.to(
                dtype=next(base_model.parameters()).dtype
            )
        model = JointSDARPolicy(base_model, position_head)
    else:
        model = base_model
    policy_parameters = sum(value.numel() for value in base_model.parameters())
    policy_trainable_parameters = sum(
        value.numel() for value in base_model.parameters() if value.requires_grad
    )
    upm_parameters = (
        sum(value.numel() for value in position_head.parameters())
        if position_head is not None else 0
    )
    upm_trainable_parameters = (
        sum(
            value.numel()
            for value in position_head.parameters()
            if value.requires_grad
        )
        if position_head is not None else 0
    )
    parameter_audit = {
        "full_parameter_update": True,
        "policy_parameters": int(policy_parameters),
        "policy_trainable_parameters": int(policy_trainable_parameters),
        "upm_parameters": int(upm_parameters),
        "upm_trainable_parameters": int(upm_trainable_parameters),
        "joint_parameters": int(policy_parameters + upm_parameters),
        "joint_trainable_parameters": int(
            policy_trainable_parameters + upm_trainable_parameters
        ),
    }
    if parameter_audit["policy_parameters"] != parameter_audit[
        "policy_trainable_parameters"
    ]:
        raise ValueError("joint RL unexpectedly froze SDAR policy parameters")
    if parameter_audit["upm_parameters"] != parameter_audit[
        "upm_trainable_parameters"
    ]:
        raise ValueError("DCoLT unexpectedly froze UPM parameters")
    if hasattr(model, "config"):
        model.config.fuse_cross_entropy = False
        model.config.use_cache = False
    if bool(config.training.gradient_checkpointing_enable):
        model.gradient_checkpointing_enable()

    optimizer_config = config.optimizer.params
    no_decay = ("bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight")
    grouped_parameters = [
        {
            "params": [
                value for name, value in model.named_parameters()
                if value.requires_grad and not any(key in name for key in no_decay)
            ],
            "weight_decay": float(optimizer_config.weight_decay),
        },
        {
            "params": [
                value for name, value in model.named_parameters()
                if value.requires_grad and any(key in name for key in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if str(config.optimizer.name) != "adamw":
        raise ValueError("multitrace v1 is registered with AdamW")
    optimizer = AdamW(
        grouped_parameters,
        lr=float(optimizer_config.learning_rate),
        betas=(float(optimizer_config.beta1), float(optimizer_config.beta2)),
        eps=float(optimizer_config.epsilon),
        weight_decay=float(optimizer_config.weight_decay),
    )

    thought_path = os.path.realpath(str(config.dataset.thought_optimization_path))
    action_path = os.path.realpath(str(config.dataset.action_optimization_path))
    native_rows = build_native_rows(
        thought_path,
        action_path,
        int(tokenizer.mask_token_id),
        int(config.training.sample_order_seed),
        int(config.training.thought_block_size),
    )
    observed_methods = {row["registered_method"] for row in native_rows}
    if observed_methods != {registered_method}:
        raise ValueError(
            "registered method/data drift expected={} observed={}".format(
                registered_method, sorted(observed_methods)
            )
        )
    maximum_rows = int(config.training.get("max_native_replay_rows", 0))
    uncapped_rows = len(native_rows)
    if maximum_rows > 0 and len(native_rows) > maximum_rows:
        if maximum_rows < 2:
            raise ValueError("bounded multitrace replay must retain both sources")
        thought_limit = maximum_rows // 2
        action_limit = maximum_rows - thought_limit
        native_rows = (
            [row for row in native_rows if row["source"] == "thought"][
                :thought_limit
            ]
            + [row for row in native_rows if row["source"] == "action"][
                :action_limit
            ]
        )
        native_rows.sort(
            key=lambda row: (
                row["sample_id"], row["source"], int(row["round_id"])
            )
        )
    if not any(row["source"] == "thought" for row in native_rows) or not any(
        row["source"] == "action" for row in native_rows
    ):
        raise ValueError("bounded native replay rows must cover Thought and Action")
    unpadded_native_rows = len(native_rows)
    native_rows, padding_summary = pad_native_rows_for_distributed(
        native_rows,
        accelerator.num_processes,
        int(config.training.gradient_accumulation_steps),
    )
    dataset = NativeReplayDataset(native_rows)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=single_collate, num_workers=0
    )
    accumulation = int(config.training.gradient_accumulation_steps)
    epochs = int(config.training.num_train_epochs)
    # DataLoaderShard pads a non-divisible final shard so every process sees
    # ceil(N/world) complete rows and participates in the same collectives.
    # Optimizer steps are synchronized across ranks, hence they are counted in
    # local microbatches rather than multiplied by world size.
    step_plan = distributed_step_plan(
        len(dataset), accelerator.num_processes, accumulation, epochs
    )
    if step_plan["padded_global_microbatches_per_epoch"] != len(dataset):
        raise RuntimeError(
            "explicit distributed padding did not eliminate implicit repeats"
        )
    local_batches_per_epoch = step_plan["local_microbatches_per_epoch"]
    updates_per_epoch = step_plan["optimizer_steps_per_epoch"]
    expected_updates = step_plan["expected_optimizer_steps"]
    scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max(expected_updates, 1),
        num_warmup_steps=int(config.lr_scheduler.params.warmup_steps),
        min_lr_scale=float(config.lr_scheduler.params.min_lr_scale),
    )
    model, optimizer, scheduler, dataloader = accelerator.prepare(
        model, optimizer, scheduler, dataloader
    )
    if len(dataloader) != local_batches_per_epoch:
        raise RuntimeError(
            "distributed dataloader length drift observed={} expected={}".format(
                len(dataloader), local_batches_per_epoch
            )
        )

    # Rollout behavior scores are produced in eval mode.  DCoLT's learned
    # position head amplifies SDAR's train/eval attention-dispatch difference
    # (0.080269 in job 18454577), so its denominator and differentiable
    # numerator must remain in the behavior-policy eval coordinate. ``eval``
    # does not disable gradients.  Keep the established train coordinate for
    # MAPG/NeDA so checkpoint continuations remain contract-compatible with
    # their already accepted iterations; their measured replay drift is below
    # 5e-7.
    old_policy_scoring_mode = "eval" if registered_method == "dcolt" else "train"
    if old_policy_scoring_mode == "eval":
        model.eval()
    else:
        model.train()
    drift = {"thought": [], "action": []}
    position_drift: List[float] = []
    local_drift_coordinates: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in dataloader:
            for key in (
                "extended_input_ids", "prediction_mask", "labels", "position_ids",
                "adv_map", "rollout_logp",
            ):
                batch[key] = batch[key].to(accelerator.device)
            if batch.get("position_mask_index") is not None:
                batch["position_mask_index"] = batch[
                    "position_mask_index"
                ].to(accelerator.device)
            with exact_replay_numerics():
                if batch.get("position_decision") is not None:
                    score, entropy, position_logp = _projected_logp(
                        model,
                        batch,
                        return_entropy=True,
                        return_position=True,
                    )
                else:
                    score, entropy = _projected_logp(
                        model, batch, return_entropy=True
                    )
                    position_logp = None
            index = int(batch["index"])
            dataset.old_logp[index] = score[0].detach().float().cpu()
            dataset.row_entropy[index] = float(entropy.detach().cpu())
            if position_logp is not None:
                dataset.old_position_logp[index] = float(
                    position_logp.detach().float().cpu()
                )
                stored_position = float(
                    batch["position_decision"]["behavior_logprob"]
                )
                position_drift.append(
                    abs(dataset.old_position_logp[index] - stored_position)
                )
            selected = batch["prediction_mask"]
            error = (score - batch["rollout_logp"])[selected].abs()
            drift[str(batch["source"])].extend(float(value) for value in error.cpu())
            local_drift_coordinates.extend(
                _drift_coordinates(batch, score, accelerator.process_index)
            )
    accelerator.wait_for_everyone()
    selection_summary = {
        "mode": "all-recorded-steps",
        "active_rows": sum(bool(value) for value in dataset.active),
        "inactive_rows": sum(not bool(value) for value in dataset.active),
        "distributed_padding": dict(padding_summary),
    }
    if registered_method == "egspo":
        local_metadata = []
        for index, row in enumerate(dataset.rows):
            if math.isfinite(dataset.row_entropy[index]):
                local_metadata.append(
                    (
                        index,
                        row["sample_id"],
                        row["source"],
                        int(row["round_id"]),
                        float(dataset.row_entropy[index]),
                        int((row.get("step_selection") or {}).get("top_k", 0)),
                    )
                )
        gathered = [None] * accelerator.num_processes
        if accelerator.num_processes > 1:
            torch.distributed.all_gather_object(gathered, local_metadata)
        else:
            gathered = [local_metadata]
        by_trace: Dict[Any, List[Any]] = {}
        seen_indices = set()
        for per_rank in gathered:
            for item in per_rank or []:
                index, sample_id, source, round_id, entropy, top_k = item
                if index in seen_indices:
                    continue
                seen_indices.add(index)
                if top_k <= 0:
                    raise ValueError("EGSPO replay row lacks high-entropy top-k")
                by_trace.setdefault((sample_id, source, top_k), []).append(
                    (entropy, -round_id, index)
                )
        active = set()
        for (_, _, top_k), candidates in by_trace.items():
            candidates.sort(reverse=True)
            active.update(item[2] for item in candidates[:top_k])
        if not active:
            raise ValueError("EGSPO selected no realized diffusion step")
        dataset.active = [
            index in active
            and not bool(
                dataset.rows[index].get("is_distributed_padding", False)
            )
            for index in range(len(dataset))
        ]
        selection_summary = {
            "mode": "high_entropy",
            "active_rows": sum(bool(value) for value in dataset.active),
            "inactive_rows": sum(not bool(value) for value in dataset.active),
            "n_traces": len(by_trace),
            "active_index_sha256": sha256_json(
                [
                    index
                    for index, value in enumerate(dataset.active)
                    if bool(value)
                ]
            ),
            "distributed_padding": dict(padding_summary),
        }
    drift_summary = {}
    for source in ("thought", "action"):
        values = drift[source]
        statistics = torch.tensor(
            [
                float(len(values)),
                float(sum(values)),
                float(max(values) if values else 0.0),
            ],
            dtype=torch.float64,
            device=accelerator.device,
        )
        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(
                statistics[:2], op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                statistics[2:], op=torch.distributed.ReduceOp.MAX
            )
        count = int(statistics[0].item())
        drift_summary[source] = {
            "n": count,
            "max_abs": float(statistics[2].item()),
            "mean_abs": float(statistics[1].item()) / count if count else 0.0,
        }
    position_statistics = torch.tensor(
        [
            float(len(position_drift)),
            float(sum(position_drift)),
            float(max(position_drift) if position_drift else 0.0),
        ],
        dtype=torch.float64,
        device=accelerator.device,
    )
    if accelerator.num_processes > 1:
        torch.distributed.all_reduce(
            position_statistics[:2], op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            position_statistics[2:], op=torch.distributed.ReduceOp.MAX
        )
    position_count = int(position_statistics[0].item())
    drift_summary["position"] = {
        "n": position_count,
        "max_abs": float(position_statistics[2].item()),
        "mean_abs": (
            float(position_statistics[1].item()) / position_count
            if position_count else 0.0
        ),
    }
    diagnostic_top_k = int(config.training.get("replay_diagnostic_top_k", 64))
    local_top = _top_drift_coordinates(
        [local_drift_coordinates], diagnostic_top_k
    )
    gathered_coordinates = [None] * accelerator.num_processes
    if accelerator.num_processes > 1:
        torch.distributed.all_gather_object(gathered_coordinates, local_top)
    else:
        gathered_coordinates = [local_top]
    top_coordinates = _top_drift_coordinates(
        gathered_coordinates, diagnostic_top_k
    )
    tolerances = {
        source: float(
            config.training.get(source + "_rollout_replay_tolerance", 0.05)
        )
        for source in ("thought", "action")
    }
    failures = [
        source
        for source in ("thought", "action")
        if drift_summary[source]["max_abs"] > tolerances[source]
    ]
    position_tolerance = float(
        config.training.get("position_rollout_replay_tolerance", 0.05)
    )
    tolerances["position"] = position_tolerance
    if (
        registered_method in POSITION_RATIO_METHODS
        and (
            drift_summary["position"]["n"] == 0
            or drift_summary["position"]["max_abs"] > position_tolerance
        )
    ):
        failures.append("position")
    diagnostic_path = Path(config.experiment.project) / "replay_drift_diagnostic.json"
    diagnostic = {
        "contract_version": REPLAY_DIAGNOSTIC_CONTRACT_VERSION,
        "status": "FAIL" if failures else "PASS",
        "scientific_role": (
            "engineering-only pre-optimizer exact-path consistency diagnostic"
        ),
        "registered_method": registered_method,
        "joint_commitment_objective": (
            {
                "mapg": "separate MAPG token/position clipping",
                "dcolt": "joint DCoLT token/UPM-position clipping",
                "neda": "MAPG coordinate with Action-mediated step credit",
                "neda_no_position": (
                    "NeDA commitment-token clipping without position ratio"
                ),
                "neda_token_only": (
                    "NeDA commitment-token clipping without position or "
                    "Action credit"
                ),
            }.get(registered_method)
        ),
        "position_head_contract": (
            position_head_metadata.get("contract_version")
            if position_head_metadata else None
        ),
        "pretrained_model": os.path.realpath(pretrained_model),
        "thought_data": thought_path,
        "thought_data_sha256": sha256_file(thought_path),
        "action_data": action_path,
        "action_data_sha256": sha256_file(action_path),
        "sample_order_seed": int(config.training.sample_order_seed),
        "uncapped_native_rows": int(uncapped_rows),
        "unpadded_native_rows": int(unpadded_native_rows),
        "checked_native_rows": int(len(dataset)),
        "distributed_padding": {
            **padding_summary,
            "zero_credit": True,
            "implicit_accelerate_repeats": 0,
        },
        "world_size": int(accelerator.num_processes),
        "rollout_replay_drift": drift_summary,
        "old_policy_scoring_mode": old_policy_scoring_mode,
        "dcolt_collective_graph": (
            "all-native-rows-upm-zero-anchor-v1"
            if registered_method == "dcolt" else None
        ),
        "process_group_timeout_seconds": process_group_timeout_seconds,
        "tolerances": tolerances,
        "failed_sources": failures,
        "diagnostic_top_k": diagnostic_top_k,
        "top_coordinates": top_coordinates,
        "numerics": {
            "tf32": False,
            "sdpa_backend": "math",
            "rmsnorm": "reference-fp32-scoped",
        },
        "parameter_audit": parameter_audit,
    }
    diagnostic["receipt_sha256"] = sha256_json(diagnostic)
    if accelerator.is_main_process:
        _atomic_json(diagnostic_path, diagnostic)
    accelerator.wait_for_everyone()
    if failures:
        source = max(
            failures,
            key=lambda name: drift_summary[name]["max_abs"] / tolerances[name],
        )
        raise RuntimeError(
            "{} rollout/learner replay drift {:.6f} exceeds {:.6f}; "
            "diagnostic={}".format(
                source,
                drift_summary[source]["max_abs"],
                tolerances[source],
                diagnostic_path,
            )
        )

    # The fixed-layout GPU gate uses the exact same Accelerator/ZeRO-3 path as
    # training but stops before any parameter mutation.  A direct unwrapped
    # model probe is insufficient because hub-kernel decorators may select a
    # different forward after distributed preparation.
    if bool(config.training.get("replay_probe_only", False)):
        probe_path = Path(config.experiment.project) / "replay_probe_receipt.json"
        receipt = {
            "contract_version": "neda-v4-multitrace-replay-probe-v1",
            "status": "PASS",
            "registered_method": registered_method,
            "pretrained_model": os.path.realpath(pretrained_model),
            "thought_data_sha256": sha256_file(thought_path),
            "action_data_sha256": sha256_file(action_path),
            "uncapped_native_rows": int(uncapped_rows),
            "unpadded_native_replay_rows": int(unpadded_native_rows),
            "checked_native_rows": int(len(dataset)),
            "n_native_replay_rows": int(len(dataset)),
            "distributed_padding": {
                **padding_summary,
                "zero_credit": True,
                "implicit_accelerate_repeats": 0,
            },
            "rollout_replay_drift": drift_summary,
            "replay_diagnostic": os.path.realpath(str(diagnostic_path)),
            "replay_diagnostic_sha256": sha256_file(str(diagnostic_path)),
            "step_selection": selection_summary,
            "sample_order_seed": int(config.training.sample_order_seed),
        }
        receipt["receipt_sha256"] = sha256_json(receipt)
        if accelerator.is_main_process:
            _atomic_json(probe_path, receipt)
        accelerator.wait_for_everyone()
        logger.info(
            "DeepSpeed replay-only probe PASS thought=%g action=%g",
            drift_summary["thought"]["max_abs"],
            drift_summary["action"]["max_abs"],
        )
        return

    metrics_path = Path(config.experiment.project) / "train_metrics.jsonl"
    receipt_path = Path(config.experiment.project) / "multitrace_receipt.json"
    if accelerator.is_main_process:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("", encoding="utf-8")
    global_step = 0
    initial_ratio_error = 0.0
    source_microbatches = {"thought": 0, "action": 0}
    active_source_microbatches = {"thought": 0, "action": 0}
    # Per-update statistics are accumulated over every local microbatch and
    # reduced over every rank at the synchronization boundary.  Logging only
    # rank zero's final microbatch would not describe the optimizer update,
    # especially when its final distributed quantum contains zero-credit
    # padding rows.
    update_statistics = torch.zeros(
        10, dtype=torch.float64, device=accelerator.device
    )
    started = time.time()
    for epoch in range(epochs):
        for batch in dataloader:
            for key in (
                "extended_input_ids", "prediction_mask", "labels", "position_ids",
                "adv_map", "rollout_logp",
            ):
                batch[key] = batch[key].to(accelerator.device)
            if batch.get("position_mask_index") is not None:
                batch["position_mask_index"] = batch[
                    "position_mask_index"
                ].to(accelerator.device)
            source = str(batch["source"])
            source_microbatches[source] += 1
            active = bool(batch.get("active", True))
            if active:
                active_source_microbatches[source] += 1
            old_logp = dataset.old_logp[int(batch["index"])].unsqueeze(0).to(
                accelerator.device
            )
            selected = batch["prediction_mask"]
            joint_thought = (
                source == "thought"
                and batch.get("position_decision") is not None
                and registered_method in POSITION_RATIO_METHODS
            )
            token_ablation_thought = (
                source == "thought"
                and registered_method in NEDA_TOKEN_ABLATION_METHODS
            )
            with accelerator.accumulate(model):
                with exact_replay_numerics():
                    position_ratio_value = None
                    if joint_thought:
                        new_logp, new_position_logp = _projected_logp(
                            model, batch, return_position=True
                        )
                        old_position_logp = torch.as_tensor(
                            dataset.old_position_logp[int(batch["index"])],
                            dtype=torch.float32,
                            device=accelerator.device,
                        )
                        token_log_ratio = (
                            new_logp[selected] - old_logp[selected]
                        )
                        if registered_method in ("mapg", "neda"):
                            # MAPG's reported GSPO implementation clips token
                            # and position ratios separately.  The sequence
                            # token ratio uses a geometric mean over tokens.
                            token_ratio = token_log_ratio.mean().clamp(
                                -10.0, 10.0
                            ).exp()
                            position_ratio = (
                                new_position_logp - old_position_logp
                            ).clamp(-10.0, 10.0).exp()
                            token_clipped = token_ratio.clamp(
                                1.0 - float(config.training.eps),
                                1.0 + float(config.training.eps),
                            )
                            position_clipped = position_ratio.clamp(
                                1.0 - float(config.training.eps),
                                1.0 + float(config.training.eps),
                            )
                            credit_scalar = batch["step_credit"].to(
                                accelerator.device
                            )
                            if not active:
                                credit_scalar = torch.zeros_like(credit_scalar)
                            token_surrogate = torch.minimum(
                                token_ratio * credit_scalar,
                                token_clipped * credit_scalar,
                            )
                            position_surrogate = torch.minimum(
                                position_ratio * credit_scalar,
                                position_clipped * credit_scalar,
                            )
                            policy_loss = -(
                                token_surrogate + position_surrogate
                            )
                            metric_ratios = torch.stack(
                                [token_ratio, position_ratio]
                            )
                            position_ratio_value = position_ratio
                        else:
                            # Public DCoLT/LLaDOU code forms one trajectory
                            # ratio from token and learned-UPM log likelihoods.
                            joint_log_ratio = (
                                token_log_ratio.sum()
                                + new_position_logp
                                - old_position_logp
                            ).clamp(-10.0, 10.0)
                            joint_ratio = joint_log_ratio.exp()
                            joint_clipped = joint_ratio.clamp(
                                1.0 - float(config.training.eps),
                                1.0 + float(config.training.eps),
                            )
                            credit_scalar = batch["step_credit"].to(
                                accelerator.device
                            )
                            if not active:
                                credit_scalar = torch.zeros_like(credit_scalar)
                            policy_loss = -torch.minimum(
                                joint_ratio * credit_scalar,
                                joint_clipped * credit_scalar,
                            )
                            metric_ratios = joint_ratio.reshape(1)
                            position_ratio_value = (
                                new_position_logp - old_position_logp
                            ).clamp(-10.0, 10.0).exp()
                    elif token_ablation_thought:
                        # A clean NeDA no-position ablation retains the exact
                        # commitment-level token surrogate used by full NeDA:
                        # one geometric-mean token ratio and one scalar
                        # denoising-step credit.  Only the Plackett--Luce
                        # position term is absent.  Falling through to the
                        # generic per-token branch would change two factors at
                        # once and invalidate the component interpretation.
                        new_logp = _projected_logp(model, batch)
                        token_log_ratio = (
                            new_logp[selected] - old_logp[selected]
                        )
                        token_ratio = token_log_ratio.mean().clamp(
                            -10.0, 10.0
                        ).exp()
                        token_clipped = token_ratio.clamp(
                            1.0 - float(config.training.eps),
                            1.0 + float(config.training.eps),
                        )
                        credit_scalar = batch["step_credit"].to(
                            accelerator.device
                        )
                        if not active:
                            credit_scalar = torch.zeros_like(credit_scalar)
                        policy_loss = -torch.minimum(
                            token_ratio * credit_scalar,
                            token_clipped * credit_scalar,
                        )
                        metric_ratios = token_ratio.reshape(1)
                    else:
                        new_logp = _projected_logp(model, batch)
                        log_ratio = torch.where(
                            selected, (new_logp - old_logp).clamp(-10.0, 10.0),
                            torch.zeros_like(new_logp),
                        )
                        ratio = log_ratio.exp()
                        clipped = ratio.clamp(
                            1.0 - float(config.training.eps),
                            1.0 + float(config.training.eps),
                        )
                        start_pos = int(batch["start_pos"])
                        width = int(batch["response_width"])
                        credit = torch.cat(
                            [
                                torch.zeros(
                                    (1, start_pos), device=accelerator.device
                                ),
                                batch["adv_map"],
                            ],
                            dim=1,
                        )
                        if not active:
                            credit = torch.zeros_like(credit)
                        surrogate = torch.minimum(
                            ratio * credit, clipped * credit
                        )
                        policy_loss = -(
                            surrogate * selected
                        ).sum() / float(width)
                        metric_ratios = ratio[selected]
                    if global_step == 0 and active:
                        initial_ratio_error = max(
                            initial_ratio_error,
                            float(
                                (metric_ratios - 1.0)
                                .abs()
                                .max()
                                .detach()
                                .cpu()
                            ),
                        )
                    if not torch.isfinite(policy_loss):
                        raise RuntimeError("multitrace policy loss is non-finite")
                    if active:
                        detached_ratios = metric_ratios.detach().double()
                        update_statistics[0] += policy_loss.detach().double()
                        update_statistics[1] += 1.0
                        update_statistics[3] += (
                            detached_ratios - 1.0
                        ).abs().sum()
                        update_statistics[4] += float(
                            detached_ratios.numel()
                        )
                        update_statistics[5] += (
                            (detached_ratios < 1.0 - float(config.training.eps))
                            | (
                                detached_ratios
                                > 1.0 + float(config.training.eps)
                            )
                        ).double().sum()
                        update_statistics[6] += float(source == "action")
                        update_statistics[7] += float(joint_thought)
                        if position_ratio_value is not None:
                            update_statistics[8] += (
                                position_ratio_value.detach().double()
                            )
                            update_statistics[9] += 1.0
                    else:
                        update_statistics[2] += 1.0
                    accelerator.backward(policy_loss)
                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), float(config.training.max_grad_norm)
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                reduced_statistics = update_statistics.clone()
                if accelerator.num_processes > 1:
                    torch.distributed.all_reduce(
                        reduced_statistics, op=torch.distributed.ReduceOp.SUM
                    )
                active_rows = int(reduced_statistics[1].item())
                padding_rows = int(reduced_statistics[2].item())
                effective_rows = active_rows + padding_rows
                expected_effective_rows = (
                    accelerator.num_processes * accumulation
                )
                ratio_count = int(reduced_statistics[4].item())
                if (
                    active_rows <= 0
                    or effective_rows != expected_effective_rows
                    or ratio_count <= 0
                ):
                    raise RuntimeError(
                        "distributed update-metric coverage drift"
                    )
                position_count = int(reduced_statistics[9].item())
                metric = {
                    "train/optimizer_step": global_step,
                    "train/loss": float(
                        reduced_statistics[0].item()
                        / float(expected_effective_rows)
                    ),
                    "train/loss_active_mean": float(
                        reduced_statistics[0].item() / float(active_rows)
                    ),
                    "train/grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                    "train/ratio_abs_mean": float(
                        reduced_statistics[3].item() / float(ratio_count)
                    ),
                    "train/clip_fraction": float(
                        reduced_statistics[5].item() / float(ratio_count)
                    ),
                    "train/source_is_action": float(
                        reduced_statistics[6].item() / float(active_rows)
                    ),
                    "train/source_is_joint_thought": float(
                        reduced_statistics[7].item() / float(active_rows)
                    ),
                    "train/position_ratio": (
                        float(
                            reduced_statistics[8].item()
                            / float(position_count)
                        )
                        if position_count else 1.0
                    ),
                    "train/active_rows": active_rows,
                    "train/padding_rows": padding_rows,
                    "train/effective_batch_rows": effective_rows,
                    "train/ratio_coordinates": ratio_count,
                }
                if tracker_ready:
                    accelerator.log(metric, step=global_step)
                if accelerator.is_main_process:
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metric, sort_keys=True) + "\n")
                global_step += 1
                update_statistics.zero_()
                torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    ratio_tensor = torch.tensor(
        initial_ratio_error, dtype=torch.float64, device=accelerator.device
    )
    source_tensor = torch.tensor(
        [
            source_microbatches["thought"],
            source_microbatches["action"],
            active_source_microbatches["thought"],
            active_source_microbatches["action"],
        ],
        dtype=torch.long,
        device=accelerator.device,
    )
    if accelerator.num_processes > 1:
        torch.distributed.all_reduce(ratio_tensor, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(source_tensor, op=torch.distributed.ReduceOp.SUM)
    initial_ratio_error = float(ratio_tensor.item())
    source_microbatches = {
        "thought": int(source_tensor[0].item()),
        "action": int(source_tensor[1].item()),
    }
    active_source_microbatches = {
        "thought": int(source_tensor[2].item()),
        "action": int(source_tensor[3].item()),
    }
    engine_steps = getattr(model, "global_steps", None)
    if global_step != expected_updates:
        raise RuntimeError(
            "multitrace optimizer-step drift observed={} expected={}".format(
                global_step, expected_updates
            )
        )
    if (
        accelerator.distributed_type == DistributedType.DEEPSPEED
        and engine_steps is not None
        and int(engine_steps) != expected_updates
    ):
        raise RuntimeError("DeepSpeed multitrace global-step drift")
    if initial_ratio_error > 1e-6:
        raise RuntimeError("frozen-old initial ratio is not one")

    save_checkpoint(model, tokenizer, config, accelerator, str(config.model.optimized_name))
    receipt = {
        "contract_version": MULTITRACE_CONTRACT_VERSION,
        "status": "PASS",
        "thought_data": thought_path,
        "thought_data_sha256": sha256_file(thought_path),
        "action_data": action_path,
        "action_data_sha256": sha256_file(action_path),
        "behavior_checkpoint": pretrained_model,
        "registered_method": registered_method,
        "joint_commitment_objective": (
            {
                "mapg": "separate MAPG token/position clipping",
                "dcolt": "joint DCoLT token/UPM-position clipping",
                "neda": "MAPG coordinate with Action-mediated step credit",
                "neda_no_position": (
                    "NeDA commitment-token clipping without position ratio"
                ),
                "neda_token_only": (
                    "NeDA commitment-token clipping without position or "
                    "Action credit"
                ),
            }.get(registered_method)
        ),
        "position_head_contract": (
            position_head_metadata.get("contract_version")
            if position_head_metadata else None
        ),
        "uncapped_native_replay_rows": uncapped_rows,
        "unpadded_native_replay_rows": unpadded_native_rows,
        "n_native_replay_rows": len(dataset),
        "distributed_padding": {
            **padding_summary,
            "zero_credit": True,
            "implicit_accelerate_repeats": 0,
        },
        "metric_aggregation": {
            "contract_version": "neda-update-metrics-v1",
            "scope": "all-ranks-all-accumulation-microbatches",
            "effective_batch_rows": int(
                accelerator.num_processes * accumulation
            ),
            "padding_excluded_from_active_means": True,
        },
        "world_size": accelerator.num_processes,
        "local_microbatches_per_epoch": local_batches_per_epoch,
        "padded_global_microbatches_per_epoch": step_plan[
            "padded_global_microbatches_per_epoch"
        ],
        "source_microbatches": source_microbatches,
        "active_source_microbatches": active_source_microbatches,
        "step_selection": selection_summary,
        "sample_order_sha256": sha256_json(
            [[row["sample_id"], row["source"], row["round_id"]] for row in native_rows]
        ),
        "rollout_replay_drift": drift_summary,
        "old_policy_scoring_mode": old_policy_scoring_mode,
        "dcolt_collective_graph": (
            "all-native-rows-upm-zero-anchor-v1"
            if registered_method == "dcolt" else None
        ),
        "replay_diagnostic": os.path.realpath(str(diagnostic_path)),
        "replay_diagnostic_sha256": sha256_file(str(diagnostic_path)),
        "initial_ratio_max_abs_error": initial_ratio_error,
        "optimizer": "adamw-one-shared-state",
        "parameter_audit": parameter_audit,
        "expected_optimizer_steps": expected_updates,
        "observed_optimizer_steps": global_step,
        "deepspeed_global_steps": None if engine_steps is None else int(engine_steps),
        "elapsed_seconds": time.time() - started,
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    if accelerator.is_main_process:
        with receipt_path.open("w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
