"""R002: SDAR-8B zero-shot ALFWorld (ReAct) — in-process driver.

Stage 1 (this file): unconstrained ReAct loop, end-to-end.
  env (alfworld get_environment) <-> SDAR (block_diffusion_generate) <-> ReAct parse <-> metrics
Stage 2 (next): inject E004 Trie constraint at the action span (the `constraint`
  hook in block_diffusion_generate is already wired; --constrained will use it).

Run via: bash ~/yinuo_dLLM/run_r002.sh   (sets env, paths, args)

Metrics reported:
  - success rate (task done)
  - progress rate (partial reward at episode end)
  - LEGALITY rate of RAW parsed actions, measured BEFORE any difflib snapping
    (the audited DiffuAgent grounding_acc is post-snap and inflated; we report pre-snap)
"""
import os
import sys
import re
import json
import argparse
import difflib
import random
import time

# When this file is executed as a script, the joint decoder imports the shared
# generation helpers by their module name.  Reuse this already-loaded module
# instead of importing a second copy under ``r002_alfworld``.
sys.modules.setdefault("r002_alfworld", sys.modules[__name__])

# Must be present before the target Python process imports/initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

# RAIDEN alpha-torch shim: liger_kernel (used by SDAR MLP) references
# torch.distributed.tensor.DTensor, which lives at torch.distributed._tensor in
# NGC 2410's alpha torch. Point the expected name at the real class so liger's
# isinstance(...) check works instead of AttributeError. (README §7.1)
try:
    import torch.distributed.tensor as _dt  # noqa
    if not hasattr(_dt, "DTensor"):
        from torch.distributed._tensor import DTensor as _DTensor
        _dt.DTensor = _DTensor
except Exception:
    try:
        import types as _types
        import torch.distributed as _dist
        from torch.distributed._tensor import DTensor as _DTensor
        _mod = _types.ModuleType("torch.distributed.tensor")
        _mod.DTensor = _DTensor
        _dist.tensor = _mod
        import sys as _sys
        _sys.modules["torch.distributed.tensor"] = _mod
    except Exception:
        pass

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.cache_utils import DynamicCache
from models import SDARForCausalLM

# E004 grammar (neda_grammar lives in this repo root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neda_grammar import TokenTrie, ActionConstraint, LegalityMeter  # noqa
from neda_data_contract import (
    EXACT_ACTION_SCORING_LAYOUT,
    STRUCTURAL_ACTION_DELIMITER,
    build_structural_action_prefix,
    make_sampling_spec,
    make_decision_trace,
    token_boundary_at_char,
    token_spans,
    trim_generation_trace,
)
from neda_torch_replay import exact_replay_numerics, make_basic_block_attention
from neda_freeze_splits import scan_games
from neda_joint_policy import JOINT_METHODS, load_dcolt_head
from neda_repro import (
    build_model_identity,
    canonical_game_id,
    check_game_ids,
    load_split_manifest,
    order_game_files_by_manifest,
    seed_everything,
    sha256_file,
    stable_seed,
)


# =============================================================================
# SDAR block-diffusion generation (copied from SDAR/SDAR/generate.py, +constraint hook)
# =============================================================================
def _sample(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]
    vocab = logits.shape[-1]
    # Sampling and the stored behavior score must describe the same transformed
    # distribution.  BF16 softmax rounded most high-confidence Action scores to
    # exactly 1 (logp=0), so all probability work is explicitly FP32.
    logits = logits.reshape(-1, vocab).float()
    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        v, _ = torch.topk(logits, top_k)
        logits = torch.where(logits < v[..., -1, None], torch.full_like(logits, float('-inf')), logits)
    if top_p < 1.0:
        sl, si = torch.sort(logits, descending=True)
        cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
        m = cp > top_p
        m[..., 1:] = m[..., :-1].clone()
        m[..., 0] = False
        mi = torch.scatter(torch.full_like(logits, False, dtype=torch.bool), -1, si, m)
        logits = logits.masked_fill(mi, float('-inf'))
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    tok = torch.multinomial(probs, num_samples=1)
    p = torch.gather(probs, -1, tok)
    logp = torch.gather(log_probs, -1, tok)
    return tok.view(*orig_shape), p.view(*orig_shape), logp.view(*orig_shape)


def _num_transfer(block_length, steps):
    base = block_length // steps
    rem = block_length % steps
    n = torch.zeros(steps, dtype=torch.int64) + base
    n[:rem] += 1
    return n


@torch.no_grad()
def block_diffusion_generate(model, input_ids, mask_id, gen_length=256, block_length=4,
                             denoising_steps=4, temperature=1.0, top_k=0, top_p=1.0,
                             confidence_threshold=0.85, stop_ids=None, constraint=None,
                             constraint_name="none",
                             return_trace=False, trace_contract=False,
                             stop_sequences=None):
    """Returns generated token ids (full sequence incl. prompt).

    `constraint`: optional E004 callable(committed_gen_ids: list[int]) ->
    Optional[set[int]] of allowed next-token ids. Used with block_length=1 to
    decode the action AR-style left-to-right under the admissible-command Trie
    (E004 decision 3). Returns:
      - a non-empty set -> mask logits to those ids for the next position
      - empty set       -> no legal continuation (command complete) -> stop
      - None            -> position unconstrained (free)

    `return_trace`(NeDA): if True, also return per-response-token denoising trace
    for nested credit assignment. Returns (x[0], step_map, conf) where step_map[j]
    = 去噪 round index at which response token j was committed(越早揭开=越小,
    未揭开=-1),conf[j] = 该 token 被 commit 时的采样置信度。用于 rollout 构造
    per-token advantage(B^den 势能)。`trace_contract=True` additionally returns
    exact behavior log-probabilities in a versioned dict; the legacy 3-tuple is
    retained when it is False. `stop_sequences` stops after the first completed
    block containing a token subsequence(e.g. ``Action:`` for two-stage decode).
    默认 False → 行为与 eval 完全一致。
    """
    model.eval()
    sampling = make_sampling_spec(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        constraint=constraint_name if constraint is not None else "none",
    )

    def trace_payload():
        return {
            "step_map": commit_round[prompt_length:].cpu(),
            "commit_confidence": commit_conf[prompt_length:].cpu(),
            "behavior_logprobs": commit_logprob[prompt_length:].cpu(),
            "sampling": sampling,
        }

    prompt_length = input_ids.shape[1]
    pkv = DynamicCache()
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total = num_blocks * block_length
    bm = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    # The canonical local SDAR forward used by the v2 trace contract expects a
    # block mask with an explicit head/broadcast dimension: [B, 1, Q, K].  The
    # previous dynamic-module forward happened to accept [B, Q, K], which hid a
    # rollout/learner interface mismatch until contract-v2 unified both paths on
    # ``models.SDARForCausalLM``.
    attn = (
        bm.repeat_interleave(block_length, 0)
        .repeat_interleave(block_length, 1)
        .unsqueeze(0)
        .unsqueeze(1)
    )
    pos = torch.arange(total, device=model.device).unsqueeze(0)
    x = torch.full((1, total), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    # NeDA trace:每个位置在第几去噪 round 被 commit + 当时置信度(-1/0 = 未揭开)
    commit_round = torch.full((total,), -1, dtype=torch.long, device=model.device)
    commit_conf  = torch.zeros((total,), device=model.device)
    commit_logprob = torch.full((total,), -torch.inf, device=model.device)
    rnd = 0
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length
    if prefill_length > 0:
        model(x[:, :prefill_length], attention_mask=attn[:, :, :prefill_length, :prefill_length],
              position_ids=pos[:, :prefill_length], past_key_values=pkv, use_cache=True, store_kv=True)
    nt = _num_transfer(block_length, denoising_steps)
    for nb in range(prefill_blocks, num_blocks):
        cur = x[:, nb*block_length:(nb+1)*block_length].clone()
        camask = attn[:, :, nb*block_length:(nb+1)*block_length, :(nb+1)*block_length]
        cpos = pos[:, nb*block_length:(nb+1)*block_length]
        for step in range(denoising_steps + 1):
            midx = (cur == mask_id)
            if midx.sum() == 0:
                model(cur, attention_mask=camask, position_ids=cpos,
                      past_key_values=pkv, use_cache=True, store_kv=True)
                break
            logits = model(cur, attention_mask=camask, position_ids=cpos,
                           past_key_values=pkv, use_cache=True, store_kv=False).logits
            # E004 constraint: mask logits to Trie-allowed tokens (action span).
            # Used with block_length=1 so this single masked position is the
            # leftmost uncommitted action token (left-to-right, decision 3).
            if constraint is not None:
                committed = x[0, prompt_length:nb*block_length]
                committed = committed[committed != mask_id].tolist()
                allowed = constraint(committed)
                if allowed is not None:
                    if len(allowed) == 0:
                        # no legal continuation -> command complete, stop generation
                        if return_trace:
                            if trace_contract:
                                return x[0], trace_payload()
                            return x[0], commit_round[prompt_length:].cpu(), commit_conf[prompt_length:].cpu()
                        return x[0]
                    vmask = torch.full((logits.shape[-1],), float('-inf'), device=logits.device)
                    vmask[torch.tensor(sorted(allowed), device=logits.device)] = 0.0
                    logits = logits + vmask  # broadcast over [1, block_length, vocab]
            x0, x0p, x0logp = _sample(logits, temperature, top_k, top_p)
            conf = torch.where(midx, x0p, -torch.inf)
            ti = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(conf.shape[0]):
                hi = conf[j] > confidence_threshold
                if hi.sum() >= nt[step]:
                    ti[j] = hi
                else:
                    _, idx = torch.topk(conf[j], nt[step])
                    ti[j, idx] = True
            if return_trace:
                newly = ti[0].nonzero(as_tuple=True)[0]     # 本 round 新 commit 的块内局部位置
                if newly.numel() > 0:
                    g = nb * block_length + newly           # 映射到全局位置
                    commit_round[g] = rnd
                    commit_conf[g]  = conf[0][newly].detach().float()
                    commit_logprob[g] = x0logp[0][newly].detach().float()
                rnd += 1
            cur[ti] = x0[ti]
        x[:, nb*block_length:(nb+1)*block_length] = cur
        generated = x[0, prompt_length:].tolist()
        hit_stop_id = stop_ids is not None and any(
            (x[0, prompt_length:] == s).any() for s in stop_ids if s is not None
        )
        hit_stop_sequence = False
        for sequence in stop_sequences or []:
            sequence = [int(token) for token in sequence]
            if sequence and any(
                generated[start:start + len(sequence)] == sequence
                for start in range(max(0, len(generated) - len(sequence) + 1))
            ):
                hit_stop_sequence = True
                break
        if hit_stop_id or hit_stop_sequence:
            break
    if return_trace:
        if trace_contract:
            return x[0], trace_payload()
        return x[0], commit_round[prompt_length:].cpu(), commit_conf[prompt_length:].cpu()
    return x[0]


@torch.no_grad()
def exact_ar_action_generate(
    model,
    input_ids,
    mask_id,
    gen_length=24,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    stop_ids=None,
    constraint=None,
    constraint_name="none",
    stop_sequences=None,
):
    """Sample a causal Action in the exact duplicated learner layout.

    Every token is sampled from the same fixed-width, no-cache tensor layout
    used by ``exact_replay``.  Future response slots are masks and invisible to
    the current block.  Keeping the width fixed is essential: changing GEMM or
    SDPA shapes between rollout and replay caused large score drift precisely
    on uncertain Action tokens.
    """

    model.eval()
    response_width = int(gen_length)
    if response_width <= 0:
        raise ValueError("exact AR Action generation requires gen_length > 0")
    sampling = make_sampling_spec(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        constraint=constraint_name if constraint is not None else "none",
    )
    prefix_length = int(input_ids.shape[1])
    sequence = torch.full(
        (1, prefix_length + response_width),
        int(mask_id),
        dtype=torch.long,
        device=model.device,
    )
    sequence[:, :prefix_length] = input_ids
    commit_round = torch.full(
        (response_width,), -1, dtype=torch.long, device=model.device
    )
    commit_confidence = torch.zeros(
        (response_width,), dtype=torch.float32, device=model.device
    )
    behavior_logprobs = torch.full(
        (response_width,), -torch.inf, dtype=torch.float32, device=model.device
    )
    total_length = prefix_length + 2 * response_width
    attention = make_basic_block_attention(
        total_length, prefix_length, block_size=1, device=model.device
    )
    base_positions = torch.arange(
        prefix_length + response_width, dtype=torch.long, device=model.device
    ).unsqueeze(0)
    position_ids = torch.cat(
        [base_positions, base_positions[:, prefix_length:]], dim=1
    )

    for offset in range(response_width):
        committed = sequence[0, prefix_length : prefix_length + offset].tolist()
        allowed = constraint(committed) if constraint is not None else None
        if allowed is not None and len(allowed) == 0:
            break

        # At sampling time both response copies contain the exact committed
        # prefix followed by masks.  The canonical block mask lets the current
        # duplicate query see only the prefix, prior committed Action tokens,
        # and itself.
        extended = torch.cat([sequence, sequence[:, prefix_length:]], dim=1)
        with exact_replay_numerics():
            logits = model(
                input_ids=extended,
                attention_mask=attention,
                position_ids=position_ids,
                use_cache=False,
                store_kv=False,
            ).logits
        query_position = prefix_length + response_width + offset
        logits = logits[:, query_position : query_position + 1, :]
        if allowed is not None:
            vocabulary_mask = torch.full(
                (logits.shape[-1],), float("-inf"), device=logits.device
            )
            vocabulary_mask[
                torch.as_tensor(sorted(allowed), dtype=torch.long, device=logits.device)
            ] = 0.0
            logits = logits + vocabulary_mask
        token, confidence, logp = _sample(
            logits, temperature=temperature, top_k=top_k, top_p=top_p
        )
        token_id = int(token[0, 0])
        sequence[0, prefix_length + offset] = token_id
        commit_round[offset] = offset
        commit_confidence[offset] = confidence[0, 0].detach().float()
        behavior_logprobs[offset] = logp[0, 0].detach().float()

        generated = sequence[
            0, prefix_length : prefix_length + offset + 1
        ].tolist()
        if stop_ids is not None and token_id in {
            int(value) for value in stop_ids if value is not None
        }:
            break
        if any(
            list(stop_sequence)
            and len(generated) >= len(stop_sequence)
            and generated[-len(stop_sequence) :] == [int(x) for x in stop_sequence]
            for stop_sequence in (stop_sequences or [])
        ):
            break

    return sequence[0], {
        "step_map": commit_round.cpu(),
        "commit_confidence": commit_confidence.cpu(),
        "behavior_logprobs": behavior_logprobs.cpu(),
        "sampling": sampling,
        "replay_width": response_width,
        "scoring_layout": EXACT_ACTION_SCORING_LAYOUT,
    }


# =============================================================================
# ReAct prompt + parsing
# =============================================================================
TASK_KEYWORDS = [("put two", "puttwo"), ("examine", "examine"), ("look at", "examine"),
                 ("clean", "clean"), ("heat", "heat"), ("cool", "cool"), ("put", "put")]


def task_type(goal):
    g = goal.lower()
    for kw, t in TASK_KEYWORDS:
        if kw in g:
            return t
    return "put"


def build_prompt(tokenizer, prompts, goal, history):
    ex = prompts["examples"][task_type(goal)]
    ex_text = "\n".join(ex) if isinstance(ex, list) else str(ex)
    instr = prompts["instruction"]
    sys_msg = prompts.get("system_msg", "You are a helpful assistant.")
    user = (f"{instr}\n\n{ex_text}\n\nHere is the task:\n{goal}\n\n"
            + "\n".join(history)
            + "\nNow give the next step in the format 'Thought: ...\\nAction: ...'.")
    messages = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return text


ACTION_RE = re.compile(r"Action:\s*(.*)", re.IGNORECASE)


def parse_action(text):
    # take text after the last 'assistant' turn, first Action: line
    m = ACTION_RE.search(text)
    if not m:
        return None
    act = m.group(1).strip().split("\n")[0].strip()
    if act.endswith("."):
        act = act[:-1].strip()
    return act or None


def build_action_trie(tokenizer, admissible):
    """Tokenize each admissible command and build the E004 TokenTrie."""
    seqs = []
    for cmd in admissible:
        ids = tokenizer(cmd, add_special_tokens=False)["input_ids"]
        if ids:
            seqs.append(ids)
    return TokenTrie(seqs)


def constrained_action_decode(model, tokenizer, prefix_text, admissible, mask_id,
                              temperature=0.3, max_action_tokens=24):
    """E004: decode ONE action under the admissible-command Trie (token-level,
    left-to-right). Returns a string GUARANTEED to be in `admissible`.

    prefix_text already ends with 'Action: '. We decode block_length=1 (AR) and
    mask each step's logits to the Trie's allowed continuations.
    """
    trie = build_action_trie(tokenizer, admissible)

    def allowed_fn(committed):
        nxt = trie.allowed_next(committed)
        # if the committed tokens already form a complete command and there is
        # no longer command extending it, allowed_next is empty -> stop.
        return nxt

    ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    out = block_diffusion_generate(
        model, ids, mask_id, gen_length=max_action_tokens, block_length=1,
        denoising_steps=1, temperature=temperature, confidence_threshold=0.0,
        stop_ids=None, constraint=allowed_fn)
    gen_ids = out[ids.shape[1]:].tolist()
    gen_ids = [t for t in gen_ids if t != mask_id]
    # trim to the longest prefix that is a complete admissible command
    best = ""
    for k in range(1, len(gen_ids) + 1):
        if trie.is_complete(gen_ids[:k]):
            best = tokenizer.decode(gen_ids[:k], skip_special_tokens=True).strip()
    if not best:  # fallback (shouldn't happen): decode whatever was produced
        best = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return best


def _as_list(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _slice_exact_trace(trace, end):
    end = int(end)
    result = {
        "response_ids": list(trace["response_ids"][:end]),
        "step_map": list(trace["step_map"][:end]),
        "behavior_logprobs": list(trace["behavior_logprobs"][:end]),
        "commit_confidence": list(trace["commit_confidence"][:end]),
    }
    if "sampling" in trace:
        result["sampling"] = dict(trace["sampling"])
    if "replay_width" in trace:
        result["replay_width"] = int(trace["replay_width"])
    if "scoring_layout" in trace:
        result["scoring_layout"] = str(trace["scoring_layout"])
    return result


def _stop_sequences(tokenizer, texts):
    sequences = []
    seen = set()
    for text in texts:
        ids = tuple(tokenizer(text, add_special_tokens=False)["input_ids"])
        if ids and ids not in seen:
            sequences.append(list(ids))
            seen.add(ids)
    return sequences


def two_stage_decision_decode(
    model,
    tokenizer,
    prompt_ids,
    admissible,
    mask_id,
    stop_ids,
    thought_order="ao",
    action_order="ar",
    action_grammar="none",
    gen_length=128,
    action_gen_length=24,
    block_length=4,
    denoising_steps=4,
    temperature=0.3,
):
    """Decode Thought and environment-facing Action as separate exact decisions.

    AO/AR changes only SDAR block length (``block_length`` vs 1). Grammar is a
    separate switch and is only valid for AR Action. The Thought pass stops once
    it emits ``Action:``; at most the remainder of that diffusion block is
    discarded and reported, so the final path does not generate a full action
    twice.
    """

    if thought_order not in ("ao", "ar") or action_order not in ("ao", "ar"):
        raise ValueError("thought_order/action_order must be ao or ar")
    if action_grammar not in ("none", "trie"):
        raise ValueError("action_grammar must be none or trie")
    if action_grammar == "trie" and action_order != "ar":
        raise ValueError("trie grammar requires AR/block=1 Action")
    prompt_ids = [int(token) for token in prompt_ids]
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    thought_block = block_length if thought_order == "ao" else 1
    thought_steps = denoising_steps if thought_order == "ao" else 1
    thought_out, thought_raw_trace = block_diffusion_generate(
        model,
        prompt_tensor,
        mask_id,
        gen_length=gen_length,
        block_length=thought_block,
        denoising_steps=thought_steps,
        temperature=temperature,
        confidence_threshold=0.85 if thought_order == "ao" else 0.0,
        stop_ids=stop_ids,
        return_trace=True,
        trace_contract=True,
        stop_sequences=_stop_sequences(tokenizer, ("\nAction:", "Action:")),
    )
    thought_full_trace = trim_generation_trace(
        _as_list(thought_out[len(prompt_ids):]),
        _as_list(thought_raw_trace["step_map"]),
        _as_list(thought_raw_trace["behavior_logprobs"]),
        mask_id,
        stop_ids,
        confidence=_as_list(thought_raw_trace["commit_confidence"]),
        sampling=thought_raw_trace["sampling"],
    )
    thought_full_text = tokenizer.decode(
        thought_full_trace["response_ids"], skip_special_tokens=True
    )
    structural = build_structural_action_prefix(
        tokenizer,
        prompt_ids,
        thought_full_text,
        thought_full_trace["response_ids"],
    )
    marker_injected = not structural["marker_generated"]
    thought_trace = _slice_exact_trace(
        thought_full_trace, structural["thought_end"]
    )
    action_prefix_ids = structural["action_prefix_ids"]

    action_constraint = None
    if action_grammar == "trie":
        trie = build_action_trie(tokenizer, admissible)

        def action_constraint(committed):
            return trie.allowed_next(committed)

    action_tensor = torch.tensor([action_prefix_ids], dtype=torch.long, device=model.device)
    action_block = block_length if action_order == "ao" else 1
    action_steps = denoising_steps if action_order == "ao" else 1
    if action_order == "ar":
        action_out, action_raw_trace = exact_ar_action_generate(
            model,
            action_tensor,
            mask_id,
            gen_length=action_gen_length,
            temperature=temperature,
            stop_ids=stop_ids,
            constraint=action_constraint,
            constraint_name=action_grammar,
            stop_sequences=(
                _stop_sequences(tokenizer, ("\n",))
                if action_grammar == "none"
                else None
            ),
        )
    else:
        action_out, action_raw_trace = block_diffusion_generate(
            model,
            action_tensor,
            mask_id,
            gen_length=action_gen_length,
            block_length=action_block,
            denoising_steps=action_steps,
            temperature=temperature,
            confidence_threshold=0.85,
            stop_ids=stop_ids,
            constraint=action_constraint,
            constraint_name=action_grammar,
            return_trace=True,
            trace_contract=True,
            stop_sequences=(
                _stop_sequences(tokenizer, ("\n",))
                if action_grammar == "none"
                else None
            ),
        )
    action_trace = trim_generation_trace(
        _as_list(action_out[len(action_prefix_ids):]),
        _as_list(action_raw_trace["step_map"]),
        _as_list(action_raw_trace["behavior_logprobs"]),
        mask_id,
        stop_ids,
        confidence=_as_list(action_raw_trace["commit_confidence"]),
        sampling=action_raw_trace["sampling"],
    )
    if "replay_width" in action_raw_trace:
        action_trace["replay_width"] = int(action_raw_trace["replay_width"])
        action_trace["scoring_layout"] = str(action_raw_trace["scoring_layout"])
    action_text = tokenizer.decode(action_trace["response_ids"], skip_special_tokens=True)
    if "\n" in action_text:
        action_end = token_boundary_at_char(
            tokenizer,
            action_text,
            action_trace["response_ids"],
            action_text.index("\n") + 1,
        )
        action_trace = _slice_exact_trace(action_trace, action_end)
        action_text = tokenizer.decode(action_trace["response_ids"], skip_special_tokens=True)
    raw_action = action_text.split("\n", 1)[0].strip()
    if raw_action.endswith("."):
        raw_action = raw_action[:-1].strip()
    raw_action = raw_action or "[No Action Found]"

    thought_text = tokenizer.decode(thought_trace["response_ids"], skip_special_tokens=True)
    # The delimiter is a fixed interface token sequence, not a sampled policy
    # decision.  Keep it in the human-readable response, while preserving
    # separate exact Thought and Action traces for learning.
    response_ids = (
        thought_trace["response_ids"]
        + structural["delimiter_ids"]
        + action_trace["response_ids"]
    )
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    decision_traces = {
        "thought": make_decision_trace(
            prompt_ids, thought_trace, thought_text, tokenizer, kind="thought"
        ),
        "action": make_decision_trace(
            action_prefix_ids, action_trace, action_text, tokenizer, kind="action"
        ),
    }
    discarded = len(thought_full_trace["response_ids"]) - len(thought_trace["response_ids"])
    return {
        "response_text": response_text,
        "response_ids": response_ids,
        "raw_action": raw_action,
        "decision_traces": decision_traces,
        "thought_order": thought_order,
        "action_order": action_order,
        "action_grammar": action_grammar,
        "marker_injected": marker_injected,
        "structural_action_delimiter": STRUCTURAL_ACTION_DELIMITER,
        "structural_action_delimiter_ids": structural["delimiter_ids"],
        "structural_marker_rebuilt": True,
        "discarded_thought_pass_tokens": discarded,
        "decision_boundary": {
            "contract_version": structural["boundary_contract"],
            "marker_generated": bool(structural["marker_generated"]),
            "thought_full_token_count": len(thought_full_trace["response_ids"]),
            "thought_end": int(structural["thought_end"]),
            "removed_trailing_whitespace_tokens": int(
                structural["removed_trailing_whitespace_tokens"]
            ),
            "alignment_source": structural["alignment_source"],
            "dropped_boundary_overlap_text": structural[
                "dropped_boundary_overlap_text"
            ],
            "dropped_boundary_overlap_token_index": structural[
                "dropped_boundary_overlap_token_index"
            ],
            "dropped_boundary_overlap_token_id": structural[
                "dropped_boundary_overlap_token_id"
            ],
        },
    }


def snap(action, admissible):
    """DiffuAgent-style post-hoc difflib snapping (for reference/comparison only)."""
    if action in admissible:
        return action, 1.0
    best, score = None, 0.0
    for c in admissible:
        r = difflib.SequenceMatcher(None, action, c).ratio()
        if r > score:
            best, score = c, r
    return (best, score) if score > 0.5 else (action, score)


# =============================================================================
# Main eval loop
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--alfworld_config", required=True)
    ap.add_argument("--prompt_json", required=True)
    ap.add_argument("--num_games", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0,
                    help="start index within the canonical frozen split")
    ap.add_argument("--max_steps", type=int, default=30)  # Bitter Lesson uses 30
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--block_length", type=int, default=4)
    ap.add_argument("--denoising_steps", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--constrained", action="store_true", help="(stage 2) enable E004 Trie constraint")
    ap.add_argument("--decision-decode", choices=("joint", "two_stage"), default="joint",
                    help="two_stage separates Thought and Action decisions(E102)")
    ap.add_argument("--thought-order", choices=("ao", "ar"), default="ao")
    ap.add_argument("--action-order", choices=("ao", "ar"), default="ao")
    ap.add_argument("--action-grammar", choices=("none", "trie"), default="none",
                    help="grammar is orthogonal to Action order; trie requires action-order=ar")
    ap.add_argument("--execution-policy", choices=("snap", "raw"), default="snap",
                    help="mapping from raw proposal to environment action when grammar=none")
    ap.add_argument("--record-trace", action="store_true",
                    help="store exact IDs/commit path/logprobs for diagnosis")
    ap.add_argument("--eval-seed", type=int, default=0,
                    help="paired eval seed; expanded independently per game and turn")
    ap.add_argument("--split-manifest", help="frozen E108 split manifest")
    ap.add_argument("--split-name", choices=("dev_seen", "final_unseen"), default="dev_seen")
    ap.add_argument("--split-root", help="local root override; IDs/hash must match manifest")
    ap.add_argument("--allow-partial-final", action="store_true",
                    help="allow smoke testing fewer than all locked final games")
    ap.add_argument("--rl-method", choices=JOINT_METHODS, default=None,
                    help="evaluate the learned MAPG/DCoLT/NeDA Thought position policy")
    ap.add_argument("--position-temperature", type=float, default=0.5)
    ap.add_argument("--dcolt-head-path",
                    help="trained DCoLT UPM sidecar; required only for rl-method=dcolt")
    ap.add_argument("--out", default="r002_results.json")
    args = ap.parse_args()
    if args.num_games <= 0 or args.offset < 0:
        ap.error("--num_games must be positive and --offset non-negative")
    if args.rl_method == "dcolt" and not args.dcolt_head_path:
        ap.error("--rl-method dcolt requires --dcolt-head-path")
    if args.rl_method != "dcolt" and args.dcolt_head_path:
        ap.error("--dcolt-head-path is only valid for --rl-method dcolt")

    print(f"[R002] loading SDAR from {args.model_dir}")
    model = SDARForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
    model_identity = build_model_identity(args.model_dir, SDARForCausalLM)
    print("[R002] model_identity={}".format(model_identity["identity_sha256"]), flush=True)
    position_head = None
    dcolt_head_metadata = None
    dcolt_head_sha256 = None
    if args.rl_method == "dcolt":
        position_head, dcolt_head_metadata = load_dcolt_head(
            model.config, args.dcolt_head_path, map_location="cpu"
        )
        position_head = position_head.to(
            device=model.device, dtype=next(model.parameters()).dtype
        ).eval()
        dcolt_head_sha256 = sha256_file(args.dcolt_head_path)
    joint_decoder = None
    if args.rl_method is not None:
        from neda_v4_decision import (
            two_stage_decision_decode as joint_two_stage_decision_decode,
        )
        joint_decoder = joint_two_stage_decision_decode
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    mask_id = tok("<|MASK|>", add_special_tokens=False)["input_ids"]
    mask_id = mask_id[0] if isinstance(mask_id, list) else mask_id
    gen_cfg = GenerationConfig.from_pretrained(args.model_dir)
    eos = gen_cfg.eos_token_id
    stop_ids = eos if isinstance(eos, list) else [eos]
    print(f"[R002] mask_id={mask_id} eos={stop_ids}")

    import yaml
    import alfworld.agents.environment as envs
    config = yaml.safe_load(open(args.alfworld_config))
    EnvClass = envs.get_environment(config["env"]["type"])
    env_split = "eval_in_distribution" if args.split_name == "dev_seen" else "eval_out_of_distribution"
    if args.split_manifest:
        manifest = load_split_manifest(args.split_manifest)
        split_spec = manifest["splits"][args.split_name]
        split_root = args.split_root or split_spec["source_root_hint"]
        if args.split_name == "dev_seen":
            config["dataset"]["eval_id_data_path"] = split_root
        else:
            config["dataset"]["eval_ood_data_path"] = split_root
    else:
        manifest, split_spec, split_root = None, None, None
    base_env = EnvClass(config, train_eval=env_split)
    if args.split_manifest:
        frozen_ids = [row["game_id"] for row in scan_games(split_root)]
        check_game_ids(frozen_ids, split_spec)
        base_env.game_files = order_game_files_by_manifest(base_env.game_files, frozen_ids)
        base_env.num_games = len(base_env.game_files)
        if (args.split_name == "final_unseen" and args.num_games < len(frozen_ids)
                and not args.allow_partial_final):
            raise SystemExit(
                "locked final_unseen requires all {} games; pass --allow-partial-final only for smoke"
                .format(len(frozen_ids))
            )
        print("[R002] frozen split={} n={} hash={}".format(
            args.split_name, len(frozen_ids), split_spec["game_ids_sha256"][:12]))
    available = list(base_env.game_files)
    if args.offset >= len(available):
        raise SystemExit(
            "evaluation offset {} is outside {} available games".format(
                args.offset, len(available)
            )
        )
    base_env.game_files = available[
        args.offset : args.offset + args.num_games
    ]
    base_env.num_games = len(base_env.game_files)
    print(
        "[R002] evaluation shard offset={} games={}".format(
            args.offset, len(base_env.game_files)
        ),
        flush=True,
    )
    env = base_env.init_env(batch_size=1)
    prompts = json.load(open(args.prompt_json))

    meter = LegalityMeter()       # legality of RAW free action (model's unconstrained output)
    sent_meter = LegalityMeter()  # legality of the action actually SENT to the env
    results = []
    n = len(base_env.game_files)
    for gi in range(n):
        game_id = canonical_game_id(base_env.game_files[gi])
        seed_everything(stable_seed(args.eval_seed, game_id, "env-reset"))
        obs, info = env.reset()
        obs0 = obs[0]
        # goal line
        goal = ""
        for line in obs0.split("\n"):
            if "your task is to:" in line.lower():
                goal = line.split(":", 1)[1].strip()
        history = [f"Observation: {obs0.strip()}"]
        done, score, success = False, 0.0, False
        steps = 0
        traj = []   # full per-step trajectory for this game
        print(f"\n{'='*78}\n[R002] GAME {gi+1}/{n}  goal: {goal}\n{'='*78}")
        for step in range(args.max_steps):
            steps = step + 1
            adm = info.get("admissible_commands", [[]])[0]
            prompt_text = build_prompt(tok, prompts, goal, history)
            ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
            decision_seed = stable_seed(args.eval_seed, game_id, step, "decision")
            seed_everything(decision_seed)
            decision_meta = None
            decision_started = time.time()
            if args.decision_decode == "two_stage":
                decoder = joint_decoder or two_stage_decision_decode
                decoder_kwargs = {
                    "thought_order": args.thought_order,
                    "action_order": args.action_order,
                    "action_grammar": args.action_grammar,
                    "gen_length": args.gen_length,
                    "block_length": args.block_length,
                    "denoising_steps": args.denoising_steps,
                    "temperature": args.temperature,
                }
                if joint_decoder is not None:
                    decoder_kwargs.update(
                        {
                            "rl_method": args.rl_method,
                            "position_temperature": args.position_temperature,
                            "position_head": position_head,
                            # Evaluation needs the learned decision rule, not
                            # the extra teacher-forced credit diagnostic.
                            "collect_neda_action_evidence": False,
                        }
                    )
                decision_meta = decoder(
                    model,
                    tok,
                    ids[0].tolist(),
                    adm,
                    mask_id,
                    stop_ids,
                    **decoder_kwargs,
                )
                gen_text = decision_meta["response_text"]
                raw_action = decision_meta["raw_action"]
            else:
                # Legacy joint generation; optional exact trace is diagnostic only.
                if args.record_trace:
                    out, raw_trace = block_diffusion_generate(
                        model, ids, mask_id, gen_length=args.gen_length,
                        block_length=args.block_length, denoising_steps=args.denoising_steps,
                        temperature=args.temperature, confidence_threshold=0.85,
                        stop_ids=stop_ids, constraint=None, return_trace=True,
                        trace_contract=True)
                    exact = trim_generation_trace(
                        out[ids.shape[1]:].tolist(), _as_list(raw_trace["step_map"]),
                        _as_list(raw_trace["behavior_logprobs"]), mask_id, stop_ids,
                        confidence=_as_list(raw_trace["commit_confidence"]))
                    exact["sampling"] = dict(raw_trace["sampling"])
                    gen_text = tok.decode(exact["response_ids"], skip_special_tokens=True)
                    decision_meta = {"decision_traces": {"response": make_decision_trace(
                        ids[0].tolist(), exact, gen_text, tok, kind="response")}}
                else:
                    out = block_diffusion_generate(
                        model, ids, mask_id, gen_length=args.gen_length,
                        block_length=args.block_length, denoising_steps=args.denoising_steps,
                        temperature=args.temperature, confidence_threshold=0.85, stop_ids=stop_ids,
                        constraint=None)
                    gen_text = tok.decode(out[ids.shape[1]:], skip_special_tokens=True)
                raw_action = parse_action(gen_text) or "[No Action Found]"
            decision_latency = time.time() - decision_started
            # LEGALITY of the RAW (free) action, BEFORE any constraint/snapping (key metric)
            is_legal = raw_action in adm
            meter.total += 1
            meter.valid += int(is_legal)

            if args.decision_decode == "two_stage" and args.action_grammar == "trie":
                action = raw_action
                snap_score = 1.0
                action_transform = "trie"
            elif args.constrained:
                # E004: keep the free Thought, re-decode the Action under the Trie
                m = ACTION_RE.search(gen_text)
                thought = (gen_text[:m.start()] if m else gen_text).strip()
                prefix_text = prompt_text + thought + ("\n" if thought else "") + "Action: "
                action = constrained_action_decode(model, tok, prefix_text, adm, mask_id,
                                                   temperature=args.temperature)
                snap_score = 1.0  # guaranteed in admissible by construction
                gen_text = f"{thought}\nAction: {action}"  # for the trajectory log
                action_transform = "trie"
            else:
                # unconstrained baseline: difflib-snap so the episode can progress
                if args.execution_policy == "raw":
                    action, snap_score = raw_action, float(raw_action in adm)
                    action_transform = "identity"
                else:
                    action, snap_score = snap(raw_action, adm)
                    action_transform = "identity" if action == raw_action else "snap"
            sent_is_legal = action in adm
            sent_meter.total += 1
            sent_meter.valid += int(sent_is_legal)
            obs, score_t, done_t, info = env.step([action])
            obs0 = obs[0]
            score = score_t[0] if isinstance(score_t, (list, tuple)) else score_t
            done = done_t[0] if isinstance(done_t, (list, tuple)) else done_t

            # ---- live print (model generation + env response) ----
            print(f"\n--- step {steps} ---")
            print(f"[gen] {gen_text.strip()[:500]}")
            print(f"[raw_action] {raw_action!r}  legal={is_legal}  "
                  f"-> sent {action!r} (sent_legal={sent_is_legal})"
                  + ("" if action == raw_action else f" [{'constrained' if args.constrained else f'snapped {snap_score:.2f}'}]"))
            print(f"[obs] {obs0.strip()[:300]}")
            print(f"[reward] {float(score):.3f}  done={bool(done)}")

            # ---- record full trajectory ----
            traj.append({"step": steps, "admissible_n": len(adm), "generation": gen_text.strip(),
                         "raw_action": raw_action, "raw_is_legal": bool(is_legal),
                         "sent_action": action, "sent_is_legal": bool(sent_is_legal),
                         "executed_action": action, "action_transform": action_transform,
                         "decision_seed": decision_seed,
                         "decision_latency_seconds": decision_latency,
                         "snap_score": float(snap_score),
                         "observation": obs0.strip(), "reward": float(score), "done": bool(done),
                         **({"decision_traces": decision_meta.get("decision_traces", {}),
                             "decision_boundary": decision_meta.get("decision_boundary")}
                            if decision_meta is not None else {})})

            # State consistency:the observation was caused by the executed action.
            history.append(f"Thought/Action: {action}")
            history.append(f"Observation: {obs0.strip()}")
            if len(history) > 20:  # keep context bounded
                history = history[:1] + history[-18:]
            if done:
                success = bool(score == 1.0) or bool(done)
                break
        results.append({"game": gi, "game_id": game_id, "eval_seed": args.eval_seed,
                        "goal": goal, "success": bool(success),
                        "progress": float(score), "steps": steps, "trajectory": traj})
        print(f"\n[R002] >>> game {gi+1}/{n} DONE | success={success} progress={float(score):.2f} "
              f"steps={steps} | running legality={meter.rate():.1%}")
        # incremental save after each game (so partial runs are inspectable)
        json.dump({"partial": True, "done_games": gi + 1,
                   "trace_contract_version": "neda-trace-v2",
                   "model_identity": model_identity, "temperature": args.temperature,
                   "rl_method": args.rl_method,
                   "position_temperature": (
                       args.position_temperature if args.rl_method else None
                   ),
                   "dcolt_head_sha256": dcolt_head_sha256,
                   "results": results},
                  open(args.out, "w"), indent=2, ensure_ascii=False)

    sr = sum(r["success"] for r in results) / len(results)
    pr = sum(r["progress"] for r in results) / len(results)
    summary = {"num_games": len(results), "success_rate": sr, "progress_rate": pr,
               "raw_legality_rate": meter.rate(), "sent_legality_rate": sent_meter.rate(),
               "constrained": args.constrained, "eval_seed": args.eval_seed,
               "offset": args.offset,
               "split_name": args.split_name,
               "split_manifest": args.split_manifest,
               "decision_decode": args.decision_decode,
               "thought_order": args.thought_order, "action_order": args.action_order,
               "action_grammar": args.action_grammar,
               "execution_policy": args.execution_policy,
               "rl_method": args.rl_method,
               "position_temperature": (
                   args.position_temperature if args.rl_method else None
               ),
               "dcolt_head_path": (
                   os.path.realpath(args.dcolt_head_path)
                   if args.dcolt_head_path else None
               ),
               "dcolt_head_sha256": dcolt_head_sha256,
               "dcolt_head_contract": (
                   dcolt_head_metadata.get("contract_version")
                   if dcolt_head_metadata else None
               ),
               "temperature": args.temperature,
               "trace_contract_version": "neda-trace-v2",
               "model_identity": model_identity,
               "mean_decision_latency_seconds": (
                   sum(turn.get("decision_latency_seconds", 0.0)
                       for result in results for turn in result["trajectory"])
                   / max(1, sum(len(result["trajectory"]) for result in results))
               ),
               "results": results}
    json.dump(summary, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[R002] ==== SDAR-8B zero-shot ALFWorld ({'CONSTRAINED (E004)' if args.constrained else 'unconstrained'}) ====")
    print(f"[R002] games={len(results)} success_rate={sr:.1%} progress_rate={pr:.2f}")
    print(f"[R002] raw_legality_rate={meter.rate():.1%} (model free action)  "
          f"sent_legality_rate={sent_meter.rate():.1%} (action sent to env)")
    print(f"[R002] saved -> {args.out}")


if __name__ == "__main__":
    main()
