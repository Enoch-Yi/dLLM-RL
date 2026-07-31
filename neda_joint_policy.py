#!/usr/bin/env python3
"""Shared position-policy primitives for MAPG, DCoLT, and NeDA.

This module is intentionally independent of ALFWorld.  It defines the
Plackett--Luce commitment distribution used by MAPG/NeDA and the SDAR adapter
for DCoLT's learned unmasking-policy module (UPM).  The environment driver,
recorded-path learner, and tests all call the same functions so generation and
training cannot silently disagree about position likelihoods.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch
from torch import nn


JOINT_METHODS = ("mapg", "dcolt", "neda")
POSITION_POLICY_BY_METHOD = {
    "mapg": "mapg_logit",
    "dcolt": "dcolt_upm",
    "neda": "mapg_logit",
}
DCoLT_HEAD_CONTRACT_VERSION = "neda-dcolt-sdar-upm-v2"
POSITION_TRACE_CONTRACT_VERSION = "neda-position-commitment-v1"


def method_position_policy(method: str) -> str:
    method = str(method).lower()
    if method not in POSITION_POLICY_BY_METHOD:
        raise ValueError("method must be one of {}".format(JOINT_METHODS))
    return POSITION_POLICY_BY_METHOD[method]


def plackett_luce_logprob(
    scores: torch.Tensor,
    candidates: Sequence[int],
    selected: Sequence[int],
    temperature: float,
) -> torch.Tensor:
    """Log probability of an ordered sample without replacement.

    ``scores`` is a one-dimensional tensor indexed in the response coordinate.
    Candidate/selected indices are therefore stable across rollout and replay.
    The returned scalar remains differentiable with respect to ``scores``.
    """

    if scores.ndim != 1:
        raise ValueError("Plackett--Luce scores must be one-dimensional")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("position temperature must be finite and positive")
    remaining = [int(value) for value in candidates]
    ordered = [int(value) for value in selected]
    if not remaining or not ordered or len(ordered) > len(remaining):
        raise ValueError("invalid Plackett--Luce support")
    if len(set(remaining)) != len(remaining) or len(set(ordered)) != len(ordered):
        raise ValueError("Plackett--Luce indices must be unique")
    if any(value not in remaining for value in ordered):
        raise ValueError("selected position is outside Plackett--Luce support")
    # If the transition commits every remaining candidate, its selected set is
    # forced and the incidental order returned by multinomial sampling has no
    # effect on the denoising state.  The public DCoLT objective likewise sets
    # the final within-block position term to zero.
    if len(ordered) == len(remaining):
        return scores.new_zeros(())
    total = scores.new_zeros(())
    for position in ordered:
        support = torch.as_tensor(
            remaining, dtype=torch.long, device=scores.device
        )
        logits = scores.index_select(0, support).float() / float(temperature)
        target = remaining.index(position)
        total = total + torch.log_softmax(logits, dim=0)[target]
        remaining.pop(target)
    return total


@torch.no_grad()
def sample_plackett_luce(
    scores: torch.Tensor,
    candidates: Sequence[int],
    count: int,
    temperature: float,
) -> Tuple[List[int], float]:
    """Draw an ordered Plackett--Luce sample and return its behavior log-prob."""

    if scores.ndim != 1:
        raise ValueError("Plackett--Luce scores must be one-dimensional")
    remaining = [int(value) for value in candidates]
    count = int(count)
    if count <= 0 or count > len(remaining):
        raise ValueError("invalid number of positions to sample")
    if count == len(remaining):
        return remaining, 0.0
    selected: List[int] = []
    logprob = scores.new_zeros(())
    for _ in range(count):
        support = torch.as_tensor(
            remaining, dtype=torch.long, device=scores.device
        )
        logits = scores.index_select(0, support).float() / float(temperature)
        probabilities = torch.softmax(logits, dim=0)
        local = int(torch.multinomial(probabilities, 1, replacement=False).item())
        logprob = logprob + torch.log(probabilities[local])
        selected.append(remaining.pop(local))
    return selected, float(logprob.cpu())


def mapg_position_scores(logits: torch.Tensor) -> torch.Tensor:
    """MAPG practical position score: maximum vocabulary logit per position.

    The COLM 2026 implementation description uses the maximum model logit for
    its StepMerge position score.  Keeping the raw-logit definition here makes
    rollout and replay identical and adds no trainable parameters.
    """

    if logits.ndim < 2:
        raise ValueError("MAPG position scoring requires vocabulary logits")
    return logits.float().amax(dim=-1)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 256):
        super().__init__()
        if frequency_size % 2:
            raise ValueError("frequency size must be even")
        self.frequency_size = int(frequency_size)
        half = self.frequency_size // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32)
            / max(half - 1, 1)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(self.frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        frequencies = self.frequencies.to(
            device=timestep.device, dtype=torch.float32
        )
        values = timestep.float().reshape(-1, 1) * frequencies.reshape(1, -1)
        encoded = torch.cat([values.sin(), values.cos()], dim=-1)
        return self.projection(
            encoded.to(dtype=self.projection[0].weight.dtype)
        )


class AdaptiveLayerNormContinuous(nn.Module):
    """DCoLT/LLaDOU continuous AdaLN used before and after the UPM block."""

    def __init__(self, hidden_size: int, condition_size: int, eps: float = 1e-4):
        super().__init__()
        self.norm = nn.LayerNorm(
            int(hidden_size), elementwise_affine=False, eps=float(eps)
        )
        self.modulation = nn.Linear(
            int(condition_size), 2 * int(hidden_size)
        )
        nn.init.xavier_uniform_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(
        self, hidden_states: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        target_dtype = self.modulation.weight.dtype
        values = self.norm(hidden_states.to(dtype=target_dtype))
        scale, shift = self.modulation(
            condition.to(dtype=target_dtype)
        ).chunk(2, dim=-1)
        if values.ndim == 3 and scale.ndim == 2:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return values * (1.0 + scale) + shift


class DCoLTPositionHead(nn.Module):
    """One-block SDAR UPM adaptation of DCoLT/LLaDOU.

    DCoLT adds a learned one-transformer-block position head over frozen/base
    model hidden states, conditioned on denoising time, mask status, and the
    active semi-autoregressive block.  This adapter preserves those ingredients
    while using a PyTorch encoder block because the public DCoLT implementation
    is tied to LLaDA's private block class.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.time_embedding = SinusoidalTimeEmbedding(self.hidden_size)
        self.mask_embedding = nn.Embedding(2, self.hidden_size)
        self.block_embedding = nn.Embedding(2, self.hidden_size)
        self.norm_in = AdaptiveLayerNormContinuous(
            self.hidden_size, self.hidden_size
        )
        self.mask_head = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=int(num_attention_heads),
            dim_feedforward=int(intermediate_size),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=True,
        )
        self.norm_out = AdaptiveLayerNormContinuous(
            self.hidden_size, self.hidden_size
        )
        self.position_linear = nn.Linear(self.hidden_size, 1)
        nn.init.normal_(self.position_linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.position_linear.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        mask_index: torch.Tensor,
        current_block: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("DCoLT UPM hidden states must be [batch,length,hidden]")
        if mask_index.shape != hidden_states.shape[:2]:
            raise ValueError("DCoLT UPM mask shape drift")
        if current_block.shape != hidden_states.shape[:2]:
            raise ValueError("DCoLT UPM block shape drift")
        condition = self.time_embedding(timestep).unsqueeze(1)
        condition = condition + self.mask_embedding(mask_index.long())
        condition = condition + self.block_embedding(current_block.long())
        values = self.norm_in(hidden_states, condition)
        values = self.mask_head(values)
        values = self.norm_out(values, condition)
        return self.position_linear(values).squeeze(-1).float()


def _config_values(config: Any) -> Tuple[int, int, int]:
    return (
        int(getattr(config, "hidden_size")),
        int(getattr(config, "num_attention_heads")),
        int(getattr(config, "intermediate_size")),
    )


def build_dcolt_head(config: Any) -> DCoLTPositionHead:
    return DCoLTPositionHead(*_config_values(config))


def save_dcolt_head(
    head: DCoLTPositionHead,
    path: str,
    *,
    seed: int,
    source_model: str,
    extra: Optional[Mapping[str, Any]] = None,
    state_dict: Optional[Mapping[str, torch.Tensor]] = None,
) -> None:
    path = os.path.realpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "contract_version": DCoLT_HEAD_CONTRACT_VERSION,
        "seed": int(seed),
        "source_model": os.path.realpath(source_model),
        "architecture": {
            "hidden_size": head.hidden_size,
            "num_attention_heads": head.mask_head.self_attn.num_heads,
            "intermediate_size": head.mask_head.linear1.out_features,
            "num_head_layers": 1,
            "time_conditioning": True,
            "mask_embedding": True,
            "block_embedding": True,
            "adaptive_layer_norm": True,
        },
        "extra": dict(extra or {}),
        "state_dict": {
            key: value.detach().cpu()
            for key, value in (
                state_dict if state_dict is not None else head.state_dict()
            ).items()
        },
    }
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_dcolt_head(
    config: Any,
    path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> Tuple[DCoLTPositionHead, Dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("contract_version") != DCoLT_HEAD_CONTRACT_VERSION:
        raise ValueError("DCoLT UPM checkpoint contract drift")
    head = build_dcolt_head(config)
    expected = {
        "hidden_size": head.hidden_size,
        "num_attention_heads": head.mask_head.self_attn.num_heads,
        "intermediate_size": head.mask_head.linear1.out_features,
        "num_head_layers": 1,
        "time_conditioning": True,
        "mask_embedding": True,
        "block_embedding": True,
        "adaptive_layer_norm": True,
    }
    if payload.get("architecture") != expected:
        raise ValueError("DCoLT UPM architecture drift")
    head.load_state_dict(payload["state_dict"], strict=True)
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    return head, metadata


class JointPolicyOutput(NamedTuple):
    """Tuple-shaped output so DeepSpeed can traverse embedded tensors."""

    logits: torch.Tensor
    position_hidden_states: Optional[torch.Tensor] = None
    position_logits: Optional[torch.Tensor] = None


class JointSDARPolicy(nn.Module):
    """DeepSpeed-compatible single module containing SDAR and optional UPM."""

    def __init__(
        self,
        policy: nn.Module,
        position_head: Optional[DCoLTPositionHead] = None,
    ):
        super().__init__()
        self.policy = policy
        self.position_head = position_head

    @property
    def config(self) -> Any:
        return self.policy.config

    def gradient_checkpointing_enable(self) -> None:
        self.policy.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        store_kv: bool = False,
        position_response_start: Optional[int] = None,
        position_response_width: Optional[int] = None,
        position_timestep: Optional[torch.Tensor] = None,
        position_mask_index: Optional[torch.Tensor] = None,
        position_current_block: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> JointPolicyOutput:
        outputs = self.policy.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            store_kv=store_kv,
            **kwargs,
        )
        hidden = outputs.last_hidden_state
        logits = self.policy.lm_head(hidden)
        if self.position_head is None:
            return JointPolicyOutput(logits=logits)
        if (
            position_response_start is None
            or position_response_width is None
            or position_timestep is None
            or position_mask_index is None
            or position_current_block is None
        ):
            return JointPolicyOutput(logits=logits)
        begin = int(position_response_start) + int(position_response_width)
        end = begin + int(position_response_width)
        response_hidden = hidden[:, begin:end, :]
        position_logits = self.position_head(
            response_hidden,
            position_timestep,
            position_mask_index,
            position_current_block,
        )
        return JointPolicyOutput(
            logits=logits,
            position_hidden_states=response_hidden,
            position_logits=position_logits,
        )


def _read_model_config(model_path: str) -> SimpleNamespace:
    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return SimpleNamespace(**value)


def _main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    initialize = sub.add_parser("init")
    initialize.add_argument("--model", required=True)
    initialize.add_argument("--out", required=True)
    initialize.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "init":
        torch.manual_seed(int(args.seed))
        config = _read_model_config(args.model)
        head = build_dcolt_head(config)
        save_dcolt_head(
            head,
            args.out,
            seed=int(args.seed),
            source_model=args.model,
            extra={"initialization": "fresh-sdar-adapter"},
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "contract_version": DCoLT_HEAD_CONTRACT_VERSION,
                    "path": os.path.realpath(args.out),
                    "seed": int(args.seed),
                    "parameters": sum(value.numel() for value in head.parameters()),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    _main()
