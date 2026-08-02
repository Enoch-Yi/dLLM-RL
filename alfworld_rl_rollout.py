#!/usr/bin/env python
"""
R007 — ALFWorld RL rollout (flat baseline) for TraceRL's train/rl_sdar.py.

复用 r002_alfworld.py 的 env 环路 + 生成 + ReAct 解析,对 TRAIN games 用当前策略
做多轮 rollout,按 episode 结果给 GRPO advantage,flat 广播到该 episode 每一轮样本,
写成 rl_sdar.py 要的 temp_data:[{"prompt","response","reward"(=advantage)}, ...]。

GRPO 分组:把前 num_games 个 game 在 game_files 里各复制 k 份 → 连续 k 次 reset = 同一个
game,按 reset_index//k 分组;组内 advantage = (R_i - mean)/(std+eps)。flat = 同一 episode
的所有轮共用该 episode 的 advantage(nested CA 留给 NeDA 在这一层换)。

用法(单卡 qrsh 或被 qsub 调):
  python alfworld_rl_rollout.py --model_dir <当前策略ckpt的推理目录> \
      --alfworld_config environment/alfworld/base_config.yaml \
      --prompt_json <...>/alfworld_react.json \
      --num_games 32 --k 8 --temperature 1.0 \
      --out <project>/temp_data/rl_alfworld.json
"""
import os
import sys
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoTokenizer, GenerationConfig
from models import SDARForCausalLM

# 复用 R002 的环路组件(同一个 build_prompt / 生成 / 解析,保证与 SFT 策略一致)
from r002_alfworld import (  # noqa
    build_prompt,
    block_diffusion_generate,
    parse_action,
    snap,
)
from neda_v4_decision import two_stage_decision_decode
from neda_joint_policy import JOINT_METHODS, load_dcolt_head
from neda_credit import compute_advantage_variants
from neda_data_contract import (
    TRACE_CONTRACT_VERSION,
    history_action,
    make_decision_trace,
    make_sample_id,
    trim_generation_trace,
    validate_rollout_record,
)
from neda_repro import (
    build_model_identity,
    canonical_game_id,
    seed_everything,
    sha256_file,
    sha256_json,
    stable_seed,
)


def extract_goal(obs0):
    for line in obs0.split("\n"):
        if "your task is to:" in line.lower():
            return line.split(":", 1)[1].strip()
    return ""


def nested_adv_map(A, round_map, conf_map, lam, gamma):
    """NeDA Stage 1(nested-lite)per-token advantage = A^env + λ·B^den shaping。

    B^den 用置信度势能 φ_τ = (累计到 round τ 的置信度) / 总置信度(0→1 单调),
    去噪层 PBRS shaping:token j(在 round τ_j 被 commit)拿 λ·(γ·φ_{τ_j} − φ_{τ_j−1});
    未揭开的 token(round=-1)不加 shaping。γ=1 时逐 round shaping telescope 到常数 → 不引偏,
    只在去噪步之间【重分配】同一个 episode advantage(这就是"去噪感知信用分配")。
    返回长度 = len(round_map) 的 per-token advantage 列表。
    """
    n = len(round_map)
    committed = [(j, round_map[j], conf_map[j]) for j in range(n) if round_map[j] >= 0]
    if not committed:
        return [A] * n
    Rmax = max(r for _, r, _ in committed)
    csum = [0.0] * (Rmax + 1)          # 每 round 的置信度和
    for _, r, c in committed:
        csum[r] += max(c, 0.0)
    total = sum(csum) or 1.0
    phi = [0.0] * (Rmax + 2)           # phi[-1]=0 用 phi[Rmax+1] 之外的 0 处理;phi[t]=累计/total
    acc = 0.0
    for t in range(Rmax + 1):
        acc += csum[t]
        phi[t] = acc / total
    def sh(t):                          # γ·φ_t − φ_{t-1}
        prev = phi[t - 1] if t - 1 >= 0 else 0.0
        return gamma * phi[t] - prev
    return [A + lam * sh(round_map[j]) if round_map[j] >= 0 else A for j in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="当前策略(SFT 或上一轮 RL ckpt)的推理目录")
    ap.add_argument("--alfworld_config", required=True)
    ap.add_argument("--prompt_json", required=True)
    ap.add_argument("--num_games", type=int, default=32, help="本轮采样的 train game 数")
    ap.add_argument("--k", type=int, default=8, help="每个 game 的 rollout 条数(GRPO 组大小)")
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--block_length", type=int, default=4)
    ap.add_argument("--denoising_steps", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0, help="rollout 用高温探索,制造组内方差")
    ap.add_argument("--max_history", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0, help="打散 train game 的随机种子(避开 SFT 记住的靠前题)")
    ap.add_argument("--game_seed", type=int, default=None,
                    help="只控制 train game 选择;缺省沿用 legacy --seed")
    ap.add_argument("--rollout_seed", type=int, default=None,
                    help="只控制 policy sampling;按 game/replica/turn 派生,不受前序长度影响")
    ap.add_argument("--offset", type=int, default=0, help="打散后从第 offset 个 game 取(迭代换批)")
    ap.add_argument("--drop_zero_adv", action="store_true",
                    help="丢掉 advantage≈0 的样本(整组同奖励→无信号),减小 temp_data")
    # ---- NeDA nested credit assignment(Stage 1:nested-lite,置信度势能 B^den)----
    ap.add_argument("--nested", action="store_true",
                    help="开启 nested CA:记录去噪 trace,写 per-token adv_map(A^env + PBRS shaping)。"
                         "关闭=flat(只写标量 reward,与原 flat 一致)")
    ap.add_argument("--shape_lambda", type=float, default=1.0, help="B^den 去噪层 shaping 强度 λ")
    ap.add_argument("--shape_gamma", type=float, default=1.0, help="PBRS 折扣 γ(=1 时 telescope 到常数,不引偏)")
    # ---- NeDA Stage-2a:轮级(环境层)信用分配 —— 折扣 return + 组基线(免训 value)----
    ap.add_argument("--turn_gae", action="store_true",
                    help="开启轮级 CA:A^env_h = turn_gamma^(H-1-h)·A_flat,把 episode advantage 按"
                         "【距成功轮的距离】折扣回传(末轮拿满,早轮几何衰减)。turn_gamma=1 精确退回 flat(内置对照)。")
    ap.add_argument("--turn_gamma", type=float, default=0.95,
                    help="轮级折扣 γ(=1 退回 flat;<1 越靠近成功轮 credit 越高)")
    # ---- NeDA Stage-2a-V:记录轮结构(供 neda_env_value.py 学习 V^env + 轮级 GAE)----
    ap.add_argument("--record_turns", action="store_true",
                    help="每条 turn 样本额外写 ep_id/h/turn_reward(score 增量)/ep_len。"
                         "2a-V 用:配合 neda_env_value.py 学 V^env 再算轮级 GAE 覆盖 reward。"
                         "⚠️ 用 2a-V 时【不要】--drop_zero_adv(轮级 GAE 需完整 episode)。")
    # ---- NeDA Stage-2b:仅记录去噪 trace(step_map),不算 Stage-1 启发式 adv_map ----
    ap.add_argument("--record_trace", action="store_true",
                    help="生成走 return_trace,每条样本写 step_map(逐 token commit 轮次,-1=未揭开),"
                         "【不】写 Stage-1 的置信度 adv_map。2b 用:neda_composite_value.py 读 step_map "
                         "学 V^den 后自己写 adv_map。与 --record_turns 并用;与 --nested 互斥(nested 优先)。")
    # ---- v3 G0:decision/action path contract ----
    ap.add_argument("--exact_contract", action="store_true",
                    help="写 exact prompt/response IDs、decision spans、behavior logprobs 与 raw/executed Action")
    ap.add_argument("--decision_decode", choices=("joint", "two_stage"), default="joint")
    ap.add_argument("--thought_order", choices=("ao", "ar"), default="ao")
    ap.add_argument("--action_order", choices=("ao", "ar"), default="ao")
    ap.add_argument("--action_grammar", choices=("none", "trie"), default="none")
    ap.add_argument("--execution_policy", choices=("snap", "raw"), default="snap")
    # ---- v3 D2:同一 rollout 中物化 U0--U4 ----
    ap.add_argument("--credit_variants", action="store_true",
                    help="在同一记录上附加 U0--U4,禁止每个 variant 重采 rollout")
    ap.add_argument("--credit_variant", choices=("U0", "U1", "U2", "U3", "U4"), default="U0",
                    help="兼容 reward 字段选择;所有 variants 仍保存在 advantages")
    ap.add_argument("--credit_gamma", type=float, default=0.95)
    ap.add_argument("--mass_constant", type=float, default=None)
    ap.add_argument("--rl_method", choices=JOINT_METHODS, default=None,
                    help="启用 MAPG/DCoLT/NeDA 的概率位置策略与联合 commitment trace")
    ap.add_argument("--position_temperature", type=float, default=0.5)
    ap.add_argument("--dcolt_head_path", default=None,
                    help="DCoLT SDAR UPM sidecar；method=dcolt 时必需")
    ap.add_argument("--neda_credit_boundaries", type=int, default=4,
                    help="NeDA Action-evidence StepMerge 边界数")
    ap.add_argument("--out", required=True, help="temp_data json 输出路径")
    args = ap.parse_args()

    if args.decision_decode == "two_stage" and args.action_grammar == "trie" and args.action_order != "ar":
        ap.error("--action_grammar trie requires --action_order ar")
    game_seed = args.seed if args.game_seed is None else args.game_seed
    rollout_seed = args.seed if args.rollout_seed is None else args.rollout_seed
    seed_everything(rollout_seed)

    # ---- model / tokenizer ----
    model = SDARForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype="auto").to("cuda").eval()
    position_head = None
    dcolt_head_metadata = None
    if args.rl_method == "dcolt":
        if not args.dcolt_head_path or not os.path.isfile(args.dcolt_head_path):
            ap.error("--rl_method dcolt requires an existing --dcolt_head_path")
        position_head, dcolt_head_metadata = load_dcolt_head(
            model.config, args.dcolt_head_path, map_location="cpu"
        )
        position_head = position_head.to(
            device=model.device, dtype=next(model.parameters()).dtype
        ).eval()
    elif args.dcolt_head_path:
        ap.error("--dcolt_head_path is only valid for --rl_method dcolt")
    model_identity = build_model_identity(args.model_dir, SDARForCausalLM)
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    mask_id = tok.mask_token_id
    gen_cfg = GenerationConfig.from_pretrained(args.model_dir)
    eos = gen_cfg.eos_token_id
    stop_ids = eos if isinstance(eos, list) else [eos]
    print(f"[rollout] model={args.model_dir} mask_id={mask_id} eos={stop_ids} temp={args.temperature}", flush=True)

    # ---- ALFWorld TRAIN env;把前 num_games 个 game 各复制 k 份(连续 k reset = 同 game)----
    import yaml
    import alfworld.agents.environment as envs
    config = yaml.safe_load(open(args.alfworld_config))
    # train 游戏(3553 个)在 ~/.cache/alfworld;AgentBoard 的 ./data 只有 eval split。
    # 强制把 data_path 指到真正的 train 目录(否则 train_eval=train → 0 games)。
    alfw_root = os.path.expanduser(os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld"))
    if "dataset" in config:
        config["dataset"]["data_path"] = os.path.join(alfw_root, "json_2.1.1", "train")
    EnvClass = envs.get_environment(config["env"]["type"])
    base_env = EnvClass(config, train_eval="train")
    print(f"[rollout] train data_path={config.get('dataset',{}).get('data_path')} "
          f"-> {len(base_env.game_files)} train games", flush=True)
    n_games = min(args.num_games, len(base_env.game_files))
    # 打散 game 选择:靠前的 train game 多是 SFT 记住的(全成功→GRPO 无方差→无信号)。
    # 随机采样 → 多为 SFT 未专训的前沿题,模型 ~50% 成功 → 组内有方差 → 有梯度。
    import random as _random
    all_files = sorted(base_env.game_files)            # 确定性基线
    _random.Random(game_seed).shuffle(all_files)
    orig_files = all_files[args.offset: args.offset + n_games]   # 不同迭代换 offset/seed 取新题
    base_env.game_files = [f for f in orig_files for _ in range(args.k)]   # game0×k, game1×k, ...
    if hasattr(base_env, "num_games"):
        base_env.num_games = len(base_env.game_files)
    env = base_env.init_env(batch_size=1)
    prompts = json.load(open(args.prompt_json))

    total = len(base_env.game_files)   # = n_games * k
    samples = []           # 全部逐轮样本
    episode_meta = []      # 每条 episode:{game, reward, idx:[sample下标]}

    for ri in range(total):
        game_idx = ri // args.k
        rollout_id = ri % args.k
        game_id = canonical_game_id(orig_files[game_idx])
        group_id = "group-{}".format(sha256_json([game_id, game_seed, args.offset])[:16])
        episode_id = "episode-{}".format(
            sha256_json([group_id, rollout_id, rollout_seed])[:16])
        seed_everything(stable_seed(rollout_seed, game_id, rollout_id, "env-reset"))
        obs, info = env.reset()
        obs0 = obs[0]
        goal = extract_goal(obs0)
        history = [f"Observation: {obs0.strip()}"]
        ep_idx = []
        score, done = 0.0, False
        prev_score = 0.0
        for step in range(args.max_steps):
            adm = info.get("admissible_commands", [[]])[0]
            prompt_text = build_prompt(tok, prompts, goal, history)
            ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
            decision_seed = stable_seed(rollout_seed, game_id, rollout_id, step, "decision")
            seed_everything(decision_seed)
            round_map, conf_map, behavior_logprobs = None, None, None
            decision_traces = None
            response_ids = None
            if args.decision_decode == "two_stage":
                decision = two_stage_decision_decode(
                    model, tok, ids[0].tolist(), adm, mask_id, stop_ids,
                    thought_order=args.thought_order, action_order=args.action_order,
                    action_grammar=args.action_grammar, gen_length=args.gen_length,
                    block_length=args.block_length, denoising_steps=args.denoising_steps,
                    temperature=args.temperature, rl_method=args.rl_method,
                    position_temperature=args.position_temperature,
                    position_head=position_head,
                    neda_credit_boundaries=args.neda_credit_boundaries)
                gen_text = decision["response_text"]
                response_ids = decision["response_ids"]
                raw_action = decision["raw_action"]
                decision_traces = decision["decision_traces"]
                if "response" in decision_traces:
                    response_trace = decision_traces["response"]
                    round_map = list(response_trace["step_map"])
                    conf_map = list(response_trace["commit_confidence"])
                    behavior_logprobs = list(response_trace["behavior_logprobs"])
            elif args.nested or args.record_trace or args.exact_contract:
                out, raw_trace = block_diffusion_generate(
                    model, ids, mask_id, gen_length=args.gen_length,
                    block_length=args.block_length, denoising_steps=args.denoising_steps,
                    temperature=args.temperature, confidence_threshold=0.85,
                    stop_ids=stop_ids, constraint=None, return_trace=True,
                    trace_contract=True)
                exact = trim_generation_trace(
                    out[ids.shape[1]:].tolist(), raw_trace["step_map"].tolist(),
                    raw_trace["behavior_logprobs"].tolist(), mask_id, stop_ids,
                    confidence=raw_trace["commit_confidence"].tolist(),
                    sampling=raw_trace["sampling"])
                response_ids = exact["response_ids"]
                round_map = exact["step_map"]
                conf_map = exact["commit_confidence"]
                behavior_logprobs = exact["behavior_logprobs"]
                gen_text = tok.decode(response_ids, skip_special_tokens=True)
                raw_action = parse_action(gen_text) or "[No Action Found]"
                decision_traces = {"response": make_decision_trace(
                    ids[0].tolist(), exact, gen_text, tok, kind="response")}
            else:
                out = block_diffusion_generate(
                    model, ids, mask_id, gen_length=args.gen_length,
                    block_length=args.block_length, denoising_steps=args.denoising_steps,
                    temperature=args.temperature, confidence_threshold=0.85,
                    stop_ids=stop_ids, constraint=None)
                gen_text = tok.decode(out[ids.shape[1]:], skip_special_tokens=True).strip()
                raw_action = parse_action(gen_text) or "[No Action Found]"
            if args.decision_decode == "two_stage" and args.action_grammar == "trie":
                action, snap_score, action_transform = raw_action, 1.0, "trie"
            elif args.execution_policy == "raw":
                action, snap_score, action_transform = raw_action, float(raw_action in adm), "identity"
            else:
                action, snap_score = snap(raw_action, adm)   # 让 episode 能推进;训练仍用 raw proposal
                action_transform = "identity" if action == raw_action else "snap"
            # 记录样本:prompt + 模型【原始生成】(PPO 要对它算 logprob),reward 占位后填。
            # nested:同时存去噪 trace(round_map/conf_map),等 GRPO advantage 定了再算 adv_map。
            ep_idx.append(len(samples))
            si_cur = len(samples)
            sample_id = make_sample_id(game_id, rollout_id, step)
            sample = {"prompt": prompt_text, "response": gen_text, "reward": 0.0,
                            "game": game_idx, "step": step,
                            "ep_id": ri, "h": step, "turn_reward": 0.0, "ep_len": 0,
                            "game_id": game_id, "group_id": group_id,
                            "episode_id": episode_id, "turn_id": step,
                            "rollout_id": rollout_id, "sample_id": sample_id,
                            "game_seed": game_seed, "rollout_seed": rollout_seed,
                            "decision_seed": decision_seed,
                            "model_identity_sha256": model_identity["identity_sha256"],
                            "raw_action": raw_action, "executed_action": action,
                            "sent_is_legal": bool(action in adm),
                            "action_transform": action_transform,
                            "snap_score": float(snap_score),
                            # Frozen, compact state covariates used by the V4
                            # deployment credit heads.  They are captured
                            # before env.step(), so they describe the same
                            # decision coordinate as the recorded traces.
                            "state_before": {
                                "contract_version": "neda-online-credit-state-v1",
                                "admissible_commands": list(adm),
                                "cumulative_reward": float(prev_score),
                            },
                            "_round_map": None if round_map is None else list(round_map),
                            "_conf_map":  None if conf_map  is None else list(conf_map),
                            "_behavior_logprobs": None if behavior_logprobs is None else list(behavior_logprobs)}
            if args.exact_contract:
                sample["contract_version"] = TRACE_CONTRACT_VERSION
                sample["prompt_ids"] = ids[0].tolist()
                sample["response_ids"] = list(response_ids or [])
                sample["decision_traces"] = decision_traces or {}
                if args.decision_decode == "two_stage":
                    sample["decision_boundary"] = dict(
                        decision.get("decision_boundary", {})
                    )
                if "response" in sample["decision_traces"]:
                    rt = sample["decision_traces"]["response"]
                    sample["thought_span"] = rt["thought_span"]
                    sample["action_span"] = rt["action_span"]
                    sample["step_map"] = rt["step_map"]
                    sample["behavior_logprobs"] = rt["behavior_logprobs"]
                validate_rollout_record(sample, require_exact_trace=True)
            samples.append(sample)
            obs, score_t, done_t, info = env.step([action])
            obs0 = obs[0]
            score = score_t[0] if isinstance(score_t, (list, tuple)) else score_t
            done = done_t[0] if isinstance(done_t, (list, tuple)) else done_t
            # 逐轮 reward = 该步 score 增量(二值奖励下:除达成那轮外全 0)
            samples[si_cur]["turn_reward"] = float(score) - prev_score
            prev_score = float(score)
            # The next observation was caused by the executed, not proposed, action.
            history.append(history_action(action))
            history.append(f"Observation: {obs0.strip()}")
            if len(history) > args.max_history:
                history = history[:1] + history[-(args.max_history - 1):]
            if done:
                break
        reward = float(score)   # 成功=1.0 / 部分进度
        for si in ep_idx:       # 回填该 episode 的轮数(供轮级 GAE)
            samples[si]["ep_len"] = len(ep_idx)
            samples[si]["episode_horizon"] = len(ep_idx)
            samples[si]["episode_return"] = reward
        episode_meta.append({"game": game_idx, "reward": reward, "idx": ep_idx})
        if (ri + 1) % args.k == 0:
            grp = [e["reward"] for e in episode_meta[-args.k:]]
            print(f"[rollout] game {game_idx+1}/{n_games} rewards={grp}", flush=True)

    # ---- GRPO advantage:按 game 分组,(R_i - mean)/(std+eps),flat 广播到每轮 ----
    import statistics
    by_game = defaultdict(list)
    for em in episode_meta:
        by_game[em["game"]].append(em)
    for gi, eps in by_game.items():
        rs = [e["reward"] for e in eps]
        mu = sum(rs) / len(rs)
        sd = statistics.pstdev(rs) if len(rs) > 1 else 0.0
        for e in eps:
            adv = (e["reward"] - mu) / (sd + 1e-6) if sd > 0 else 0.0
            H = len(e["idx"])                       # 该 episode 的轮数
            for pos, si in enumerate(e["idx"]):
                # Stage-2a 轮级 CA:A^env_h = γ^(H-1-h)·A_flat(末轮 pos=H-1 拿满,早轮几何衰减)。
                # turn_gamma=1 → turn_adv==adv → 与 flat 逐条相同(内置对照)。
                turn_adv = (args.turn_gamma ** (H - 1 - pos)) * adv if args.turn_gae else adv
                samples[si]["reward"] = turn_adv
                samples[si]["group_advantage"] = adv
                # nested:A^env 定了,算 per-token adv_map(去噪层 B^den shaping)+ step_map
                if args.nested and samples[si].get("_round_map") is not None:
                    rm, cm = samples[si]["_round_map"], samples[si]["_conf_map"]
                    samples[si]["adv_map"] = nested_adv_map(turn_adv, rm, cm, args.shape_lambda, args.shape_gamma)
                    samples[si]["step_map"] = rm

    credit_metadata = None
    if args.credit_variants:
        samples, credit_metadata = compute_advantage_variants(
            samples, gamma=args.credit_gamma, mass_constant=args.mass_constant)
        for sample in samples:
            sample["reward"] = sample["advantages"][args.credit_variant]

    if args.drop_zero_adv:
        samples = [s for s in samples if abs(s["reward"]) > 1e-6]

    # ---- 写 temp_data(flat:prompt/response/reward;nested:再加 adv_map + step_map)----
    out_samples = []
    for s in samples:
        rec = {"prompt": s["prompt"], "response": s["response"], "reward": s["reward"],
               "game_id": s["game_id"], "group_id": s["group_id"],
               "episode_id": s["episode_id"], "turn_id": s["turn_id"],
               "rollout_id": s["rollout_id"], "sample_id": s["sample_id"],
               "episode_horizon": s["episode_horizon"],
               "group_advantage": s["group_advantage"],
               "game_seed": s["game_seed"], "rollout_seed": s["rollout_seed"],
               "decision_seed": s["decision_seed"],
               "model_identity_sha256": s["model_identity_sha256"],
               "raw_action": s["raw_action"], "executed_action": s["executed_action"],
               "sent_is_legal": s["sent_is_legal"],
               "action_transform": s["action_transform"], "snap_score": s["snap_score"],
               "episode_return": s["episode_return"],
               "state_before": s["state_before"]}
        if "advantages" in s:
            rec["advantages"] = s["advantages"]
            rec["credit_contract"] = s["credit_contract"]
        if args.exact_contract:
            for key in ("contract_version", "prompt_ids", "response_ids", "thought_span",
                        "action_span", "step_map", "behavior_logprobs", "decision_traces",
                        "decision_boundary"):
                if key in s:
                    rec[key] = s[key]
        if args.nested and "adv_map" in s:
            rec["adv_map"] = s["adv_map"]
            rec["step_map"] = s["step_map"]
        elif args.record_trace and s.get("_round_map") is not None:
            # 2b:只给 step_map(去噪 trace),adv_map 由 neda_composite_value.py 后置生成
            rec["step_map"] = s["_round_map"]
        if args.record_turns:   # 2a-V:轮结构(neda_env_value.py 读)
            rec["ep_id"] = s["ep_id"]; rec["h"] = s["h"]
            rec["turn_reward"] = s["turn_reward"]; rec["ep_len"] = s["ep_len"]
        out_samples.append(rec)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out_samples, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    artifact = {
        "contract_version": TRACE_CONTRACT_VERSION if args.exact_contract else "legacy",
        "data_path": args.out,
        "data_sha256": sha256_file(args.out),
        "n_records": len(out_samples),
        "game_seed": game_seed,
        "rollout_seed": rollout_seed,
        "offset": args.offset,
        "num_games": n_games,
        "group_size": args.k,
        "max_steps": args.max_steps,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "denoising_steps": args.denoising_steps,
        "decision_decode": args.decision_decode,
        "thought_order": args.thought_order,
        "action_order": args.action_order,
        "action_grammar": args.action_grammar,
        "execution_policy": args.execution_policy,
        "temperature": args.temperature,
        "model_identity": model_identity,
        "credit_variant": args.credit_variant if args.credit_variants else None,
        "credit_metadata": credit_metadata,
        "online_credit_context_contract": "neda-online-credit-state-v1",
        "rl_method": args.rl_method,
        "position_temperature": (
            args.position_temperature if args.rl_method is not None else None
        ),
        "dcolt_head_path": (
            os.path.realpath(args.dcolt_head_path)
            if args.dcolt_head_path else None
        ),
        "dcolt_head_contract": (
            dcolt_head_metadata.get("contract_version")
            if dcolt_head_metadata else None
        ),
        "dcolt_head_sha256": (
            sha256_file(args.dcolt_head_path) if args.dcolt_head_path else None
        ),
        "neda_credit_boundaries": (
            args.neda_credit_boundaries if args.rl_method == "neda" else None
        ),
        "sample_order_sha256": sha256_json([row["sample_id"] for row in out_samples]),
    }
    with open(args.out + ".manifest.json", "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    # ---- 轮级 CA 统计 ----
    if args.turn_gae:
        discs = [args.turn_gamma ** (len(e["idx"]) - 1 - pos)
                 for e in episode_meta for pos in range(len(e["idx"]))]
        rvals = [abs(s["reward"]) for s in samples if abs(s["reward"]) > 1e-9]
        print(f"[rollout] turn_gae ON γ={args.turn_gamma}: 折扣因子 mean={statistics.mean(discs):.3f} "
              f"min={min(discs):.4f}(最长 episode 首轮) max=1.000(末轮); "
              f"|A^env_h| mean={statistics.mean(rvals) if rvals else 0:.3f} n={len(rvals)}", flush=True)

    # ---- 统计 ----
    succ = sum(1 for e in episode_meta if e["reward"] == 1.0)
    nonzero_groups = sum(1 for eps in by_game.values()
                         if (statistics.pstdev([e["reward"] for e in eps]) if len(eps) > 1 else 0) > 0)
    print(f"[rollout] DONE games={n_games} k={args.k} episodes={len(episode_meta)} "
          f"success={succ}/{len(episode_meta)} ({succ/len(episode_meta):.1%}) "
          f"有信号组={nonzero_groups}/{n_games} samples={len(out_samples)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
