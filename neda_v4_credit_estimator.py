#!/usr/bin/env python3
"""Build frozen, method-native credit estimates for the NeDA V4 benchmark.

The estimator and fidelity stages are intentionally separate.  This module
loads the sealed ALFWorld/WebShop counterfactual benchmark, but only ``neda``
may use reference effects while fitting a score.  NeDA predictions are
game-group out-of-fold (OOF); all prior-method scores are deterministic
functions of the frozen behavior rollout and the method's registered credit
coordinate.  The downstream fidelity aggregator independently reloads the
references and joins them by ``reference_key``.

This is an attribution benchmark, not an online-policy result.
"""

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from neda_hierarchical_credit import _solve_linear_system, grouped_oof_ridge
from neda_hierarchical_credit_builder import (
    ACTION_TOKEN_FEATURES,
    STEP_FEATURES,
    THOUGHT_TOKEN_FEATURES,
    TURN_FEATURES,
    action_token_feature_vector,
    step_feature_vector,
    thought_token_feature_vector,
    turn_feature_vector,
)
from neda_repro import sha256_file, sha256_json, stable_seed


ESTIMATOR_CONTRACT_VERSION = "neda-v4-credit-estimator-v1"
CATALOG_CONTRACT_VERSION = "neda-v4-credit-reference-catalog-v1"
EXACT_CACHE_CONTRACT_VERSION = "neda-v4-exact-score-cache-v1"
METHODS = (
    "flat_rm",
    "agentic_vrpo",
    "justgrpo",
    "tracerl",
    "daca_grpo",
    "egspo",
    "neda",
)
LEVELS = ("turn", "step", "action_token", "thought_token")
AUDIT_CONTRACTS = {
    "alfworld": "neda-credit-alf-audit-v1",
    "webshop": "neda-webshop-credit-audit-v1",
}
FEATURE_NAMES = {
    "turn": TURN_FEATURES,
    "step": STEP_FEATURES,
    "thought_token": THOUGHT_TOKEN_FEATURES,
    "action_token": ACTION_TOKEN_FEATURES,
}
METHOD_SUPPORT = {
    # Flat-RM and VRPO do not define local causal coordinates.  Their native
    # trajectory score is nevertheless broadcast onto every audited decision
    # so the benchmark can directly diagnose that mismatch.
    "flat_rm": set(LEVELS),
    "agentic_vrpo": set(LEVELS),
    # JustGRPO has an AR token factorization.  The shared benchmark Action is
    # AR, while AO Thought denoising-step coordinates are genuinely N/A.
    "justgrpo": {"turn", "action_token"},
    "tracerl": set(LEVELS),
    "daca_grpo": {"step", "thought_token"},
    "egspo": {"step", "thought_token"},
    "neda": set(LEVELS),
}
METHOD_SEMANTICS = {
    "flat_rm": "group-normalized episode return broadcast to every audited coordinate",
    "agentic_vrpo": "within-task chosen/rejected return-rank preference broadcast within trajectory",
    "justgrpo": "group-normalized episode return on AR turn/Action-token coordinates",
    "tracerl": "group-normalized episode return on every realized diffusion trace coordinate",
    "daca_grpo": "DPS-modulated group advantage on Thought denoising-step/token coordinates",
    "egspo": "group advantage retained only on the highest-entropy audited Thought step",
    "neda": "game-group OOF counterfactual effect prediction on the recorded R/H/D/T coordinate",
}


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON object required: {}".format(path))
    return value


def _atomic_json(value: Mapping[str, Any], path: str) -> None:
    path = os.path.realpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} is non-finite".format(label))
    return result


def _verify_source(path: str, expected_sha: str) -> Dict[str, Any]:
    path = os.path.realpath(path)
    if not os.path.isfile(path):
        raise ValueError("sealed credit source is missing: {}".format(path))
    actual = sha256_file(path)
    if actual != str(expected_sha):
        raise ValueError(
            "sealed credit source SHA drift: {} expected={} actual={}".format(
                path, expected_sha, actual
            )
        )
    return _load(path)


def _episode_turn(
    base: Mapping[str, Any], episode_id: str, turn_id: int
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    episodes = {
        str(row["episode_id"]): row for row in base.get("base_episodes", [])
    }
    if str(episode_id) not in episodes:
        raise ValueError("branch anchor episode is absent from sealed base")
    episode = episodes[str(episode_id)]
    turns = {
        int(row["turn_id"]): row for row in episode.get("turns", [])
    }
    if int(turn_id) not in turns:
        raise ValueError("branch anchor turn is absent from sealed base")
    return episode, turns[int(turn_id)]


def _group_advantages(base: Mapping[str, Any]) -> Dict[str, float]:
    by_game: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for episode in base.get("base_episodes", []):
        by_game[str(episode["game_id"])].append(episode)
    result: Dict[str, float] = {}
    for episodes in by_game.values():
        outcomes = [_finite(row["return"], "episode return") for row in episodes]
        mean = statistics.mean(outcomes)
        scale = statistics.pstdev(outcomes) if len(outcomes) > 1 else 0.0
        for episode, outcome in zip(episodes, outcomes):
            result[str(episode["episode_id"])] = (
                (outcome - mean) / (scale + 1e-6) if scale > 0.0 else 0.0
            )
    return result


def _preference_scores(base: Mapping[str, Any]) -> Dict[str, float]:
    """Return a deterministic all-pairs extension of VRPO chosen/rejected rank."""

    by_game: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for episode in base.get("base_episodes", []):
        by_game[str(episode["game_id"])].append(episode)
    result: Dict[str, float] = {}
    for episodes in by_game.values():
        outcomes = [_finite(row["return"], "episode return") for row in episodes]
        unique = sorted(set(outcomes))
        if len(unique) == 1:
            values = [0.0] * len(outcomes)
        else:
            midpoint = (len(unique) - 1) / 2.0
            denominator = max(1.0, midpoint)
            values = [(unique.index(value) - midpoint) / denominator for value in outcomes]
        for episode, value in zip(episodes, values):
            result[str(episode["episode_id"])] = float(value)
    return result


def _coordinate_metadata(
    branch: Mapping[str, Any], reference: Mapping[str, Any]
) -> Dict[str, Any]:
    if str(reference["level"]) == "turn":
        return {"coordinate_id": None, "step_id": None, "positions": []}
    coordinate_id = str(reference.get("coordinate_id", ""))
    matches = [
        row
        for row in branch.get("local_references", [])
        if str(row.get("coordinate_id", "")) == coordinate_id
    ]
    if len(matches) != 1:
        raise ValueError("reference coordinate is absent/duplicated in branch")
    row = matches[0]
    if str(row.get("level")) != str(reference["level"]):
        raise ValueError("reference level differs from branch coordinate")
    return {
        "coordinate_id": coordinate_id,
        "step_id": int(row["step_id"]),
        "positions": [int(value) for value in row.get("positions", [])],
    }


def _features(
    level: str,
    episode: Mapping[str, Any],
    turn: Mapping[str, Any],
    coordinate: Mapping[str, Any],
) -> List[float]:
    if level == "turn":
        return turn_feature_vector(episode, turn)
    if level == "step":
        return step_feature_vector(
            episode, turn, int(coordinate["step_id"]), coordinate["positions"]
        )
    if level == "thought_token":
        positions = list(coordinate["positions"])
        if len(positions) != 1:
            raise ValueError("Thought-token reference must name one position")
        return thought_token_feature_vector(episode, turn, positions[0])
    if level == "action_token":
        positions = list(coordinate["positions"])
        if len(positions) != 1:
            raise ValueError("Action-token reference must name one position")
        return action_token_feature_vector(episode, turn, positions[0])
    raise ValueError("unsupported credit reference level: {}".format(level))


def load_reference_catalog(
    audit_paths: Mapping[str, str], expected_counts: Mapping[str, Tuple[int, int]]
) -> Dict[str, Any]:
    """Load and re-hash the two sealed reference matrices."""

    rows: List[Dict[str, Any]] = []
    audit_receipts = []
    source_receipts = []
    for environment in ("alfworld", "webshop"):
        audit_path = os.path.realpath(audit_paths[environment])
        audit = _load(audit_path)
        if (
            audit.get("contract_version") != AUDIT_CONTRACTS[environment]
            or audit.get("status") != "PASS"
            or audit.get("environment") != environment
        ):
            raise ValueError("{} aggregate credit audit is not PASS".format(environment))
        expected_bases, expected_branches = expected_counts[environment]
        if expected_bases and int(audit.get("n_base_tasks", -1)) != expected_bases:
            raise ValueError("{} base-task count drift".format(environment))
        if expected_branches and int(audit.get("n_branches", -1)) != expected_branches:
            raise ValueError("{} branch count drift".format(environment))

        bases: Dict[int, Dict[str, Any]] = {}
        base_receipts = []
        for receipt in audit.get("base_sources", []):
            task_index = int(receipt["task_index"])
            if task_index in bases:
                raise ValueError("duplicate sealed base task")
            bases[task_index] = _verify_source(receipt["path"], receipt["sha256"])
            base_receipts.append(("base", task_index, str(receipt["sha256"])))

        branches: Dict[str, Dict[str, Any]] = {}
        branch_receipts = []
        for receipt in audit.get("branch_sources", []):
            path = os.path.realpath(receipt["path"])
            if path in branches:
                raise ValueError("duplicate sealed branch path")
            branches[path] = _verify_source(path, receipt["sha256"])
            branch_receipts.append(
                (
                    "branch",
                    int(receipt["task_index"]),
                    int(receipt["anchor_index"]),
                    str(receipt["sha256"]),
                )
            )
        rebuilt_source_set = sha256_json(sorted(base_receipts + branch_receipts))
        if rebuilt_source_set != audit.get("source_set_sha256"):
            raise ValueError("{} sealed source-set SHA drift".format(environment))

        advantages = {
            task: _group_advantages(base) for task, base in bases.items()
        }
        preferences = {
            task: _preference_scores(base) for task, base in bases.items()
        }
        seen_environment = set()
        for reference in audit.get("reference_index", []):
            task_index = int(reference["task_index"])
            branch_path = os.path.realpath(reference["branch_path"])
            if task_index not in bases or branch_path not in branches:
                raise ValueError("reference index points outside sealed sources")
            branch = branches[branch_path]
            anchor = branch.get("anchor", {})
            if (
                str(anchor.get("anchor_id")) != str(reference["anchor_id"])
                or int(anchor.get("turn_id", -1)) != int(reference["turn_id"])
            ):
                raise ValueError("reference index/branch anchor drift")
            episode, turn = _episode_turn(
                bases[task_index], str(anchor["episode_id"]), int(anchor["turn_id"])
            )
            level = str(reference["level"])
            if level not in LEVELS:
                raise ValueError("unexpected reference level: {}".format(level))
            coordinate = _coordinate_metadata(branch, reference)
            reference_key = "{}:{}".format(environment, reference["reference_id"])
            if reference_key in seen_environment:
                raise ValueError("duplicate reference key")
            seen_environment.add(reference_key)
            feature_values = _features(level, episode, turn, coordinate)
            if len(feature_values) != len(FEATURE_NAMES[level]):
                raise ValueError("credit feature schema drift")
            trace = turn.get("decision_traces", {}).get("thought", {})
            rows.append(
                {
                    "reference_key": reference_key,
                    "reference_id": str(reference["reference_id"]),
                    "environment": environment,
                    "task_index": task_index,
                    "game_id": str(anchor["game_id"]),
                    "episode_id": str(anchor["episode_id"]),
                    "turn_id": int(anchor["turn_id"]),
                    "stratum": str(anchor.get("stratum", reference.get("stratum", ""))),
                    "level": level,
                    "coordinate_id": coordinate["coordinate_id"],
                    "step_id": coordinate["step_id"],
                    "positions": coordinate["positions"],
                    "features": feature_values,
                    "feature_sha256": sha256_json(feature_values),
                    "target": _finite(reference["mean_effect"], "reference effect"),
                    "reference_se": _finite(
                        reference["standard_error"], "reference standard error"
                    ),
                    "n_pairs": int(reference["n_pairs"]),
                    "group_advantage": advantages[task_index][str(anchor["episode_id"])],
                    "preference_score": preferences[task_index][str(anchor["episode_id"])],
                    "trace_sha256": str(trace.get("trace_sha256", "")),
                    # Runtime-only pointers are removed before the estimator is
                    # written.  They are necessary for exact DACA/EGSPO scoring.
                    "_episode": episode,
                    "_turn": turn,
                }
            )
        source_receipts.extend(base_receipts + branch_receipts)
        audit_receipts.append(
            {
                "environment": environment,
                "path": audit_path,
                "sha256": sha256_file(audit_path),
                "scientific_audit_sha256": audit.get("audit_sha256"),
                "source_set_sha256": audit.get("source_set_sha256"),
                "n_base_tasks": audit.get("n_base_tasks"),
                "n_branches": audit.get("n_branches"),
                "n_references": len(audit.get("reference_index", [])),
            }
        )
    keys = [row["reference_key"] for row in rows]
    if len(keys) != len(set(keys)) or not rows:
        raise ValueError("combined credit catalog is empty or duplicated")
    rows.sort(key=lambda row: row["reference_key"])
    reference_sha = sha256_json(
        [
            [
                row["reference_key"],
                row["target"],
                row["reference_se"],
                row["n_pairs"],
                row["feature_sha256"],
            ]
            for row in rows
        ]
    )
    return {
        "contract_version": CATALOG_CONTRACT_VERSION,
        "rows": rows,
        "n_references": len(rows),
        "reference_sha256": reference_sha,
        "audit_receipts": audit_receipts,
        "source_receipts_sha256": sha256_json(sorted(source_receipts)),
    }


def _fit_full_ridge(
    features: Sequence[Sequence[float]], targets: Sequence[float], ridge: float
) -> Dict[str, Any]:
    if not features or len(features) != len(targets):
        raise ValueError("full ridge requires aligned non-empty rows")
    width = len(features[0])
    if width == 0 or any(len(row) != width for row in features):
        raise ValueError("full ridge feature shape drift")
    means = [statistics.mean(float(row[j]) for row in features) for j in range(width)]
    scales = []
    for j in range(width):
        variance = statistics.mean(
            (float(row[j]) - means[j]) ** 2 for row in features
        )
        value = math.sqrt(variance)
        scales.append(1.0 if value < 1e-8 else value)
    x = [
        [
            (float(row[j]) - means[j]) / scales[j]
            for j in range(width)
        ]
        + [1.0]
        for row in features
    ]
    dimension = width + 1
    matrix = [[0.0] * dimension for _ in range(dimension)]
    vector = [0.0] * dimension
    for row, target in zip(x, targets):
        for left in range(dimension):
            vector[left] += row[left] * float(target)
            for right in range(dimension):
                matrix[left][right] += row[left] * row[right]
    for index in range(width):
        matrix[index][index] += float(ridge)
    return {
        "feature_mean": means,
        "feature_scale": scales,
        "weights": _solve_linear_system(matrix, vector),
        "ridge": float(ridge),
        "n_train": len(targets),
    }


def _neda_oof(
    rows: Sequence[Mapping[str, Any]], folds: int = 4, ridge: float = 10.0
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    predictions: Dict[str, float] = {}
    heads: Dict[str, Any] = {}
    for environment in ("alfworld", "webshop"):
        for level in LEVELS:
            selected = [
                row
                for row in rows
                if row["environment"] == environment and row["level"] == level
            ]
            if not selected:
                raise ValueError("NeDA head has no {} {} targets".format(environment, level))
            seed = stable_seed(92001, environment, level, "neda-v4-oof")
            result = grouped_oof_ridge(
                [row["features"] for row in selected],
                [row["target"] for row in selected],
                [row["game_id"] for row in selected],
                folds=folds,
                ridge=ridge,
                seed=seed,
            )
            for row, value in zip(selected, result["predictions"]):
                predictions[str(row["reference_key"])] = _finite(value, "OOF prediction")
            deployment = _fit_full_ridge(
                [row["features"] for row in selected],
                [row["target"] for row in selected],
                ridge,
            )
            head_key = "{}:{}".format(environment, level)
            heads[head_key] = {
                "environment": environment,
                "level": level,
                "feature_names": list(FEATURE_NAMES[level]),
                "n_features": len(FEATURE_NAMES[level]),
                "n_targets": len(selected),
                "n_game_groups": len(set(row["game_id"] for row in selected)),
                "folds": int(result["folds"]),
                "ridge": float(result["ridge"]),
                "seed": int(result["seed"]),
                "oof_r2_diagnostic": result["oof_r2"],
                "fold_assignments_sha256": sha256_json(result["fold_assignments"]),
                "prediction_sha256": result["prediction_sha256"],
                "target_sha256": sha256_json([row["target"] for row in selected]),
                "fold_map": result["fold_map"],
                "fold_rows": result["fold_rows"],
                "deployment_model": deployment,
                "deployment_role": (
                    "fit on the complete frozen credit benchmark only after OOF evaluation; "
                    "may score disjoint online-training games"
                ),
            }
            heads[head_key]["head_sha256"] = sha256_json(heads[head_key])
    if set(predictions) != {str(row["reference_key"]) for row in rows}:
        raise ValueError("NeDA OOF prediction coverage drift")
    return predictions, heads


class ExactTraceScorer:
    """Resumable exact model scoring used only by DACA and EGSPO adapters."""

    def __init__(self, model_path: str, cache_path: str):
        self.model_path = os.path.realpath(model_path)
        self.cache_path = os.path.realpath(cache_path)
        self.cache: Dict[str, Dict[str, Any]] = {}
        if os.path.isfile(self.cache_path):
            payload = _load(self.cache_path)
            if payload.get("contract_version") != EXACT_CACHE_CONTRACT_VERSION:
                raise ValueError("exact-score cache contract drift")
            if os.path.realpath(payload.get("model", "")) != self.model_path:
                raise ValueError("exact-score cache model drift")
            self.cache = dict(payload.get("scores", {}))
        try:
            import torch
            from transformers import AutoTokenizer
            from models import SDARForCausalLM
        except Exception as error:
            raise ValueError("exact DACA/EGSPO scoring dependencies unavailable: {}".format(error))
        if not torch.cuda.is_available():
            raise ValueError("exact DACA/EGSPO estimator requires CUDA")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = SDARForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        if hasattr(self.model, "config"):
            self.model.config.fuse_cross_entropy = False
            self.model.config.use_cache = False
        self.model.to(torch.device("cuda:0"))
        self.model.eval()
        self.pending = 0

    def _key(self, trace: Mapping[str, Any], step_id: int) -> str:
        trace_sha = str(trace.get("trace_sha256", ""))
        if len(trace_sha) != 64:
            trace_sha = sha256_json(
                [trace.get("prefix_ids", []), trace.get("response_ids", []), trace.get("step_map", [])]
            )
        return "{}:{}".format(trace_sha, int(step_id))

    def _flush(self) -> None:
        _atomic_json(
            {
                "contract_version": EXACT_CACHE_CONTRACT_VERSION,
                "model": self.model_path,
                "scores": self.cache,
                "n_scores": len(self.cache),
                "scores_sha256": sha256_json(self.cache),
            },
            self.cache_path,
        )
        self.pending = 0

    def score(self, trace: Mapping[str, Any], step_id: int, need_progress: bool) -> Dict[str, Any]:
        key = self._key(trace, step_id)
        row = dict(self.cache.get(key, {}))
        from neda_baseline_adapter_smoke import exact_step_score, progress_delta
        from neda_torch_replay import exact_replay_numerics

        with exact_replay_numerics():
            if "entropy" not in row:
                _, entropy, positions = exact_step_score(
                    self.model,
                    trace,
                    self.tokenizer.mask_token_id,
                    int(step_id),
                    4,
                    require_grad=False,
                )
                row["entropy"] = _finite(entropy, "exact step entropy")
                row["positions"] = [int(value) for value in positions]
            if need_progress and "progress_delta" not in row:
                if any(int(value) > int(step_id) for value in trace["step_map"]):
                    delta = progress_delta(
                        self.model, trace, self.tokenizer.mask_token_id, int(step_id)
                    )
                    terminal = False
                else:
                    # The paper's next-state DPS is undefined after the final
                    # commit.  Its registered adapter uses zero modulation at
                    # that boundary rather than inventing a future state.
                    delta = 0.0
                    terminal = True
                row["progress_delta"] = _finite(delta, "exact DPS delta")
                row["terminal_step"] = terminal
        self.cache[key] = row
        self.pending += 1
        if self.pending >= 20:
            self._flush()
        return row

    def finish(self) -> Dict[str, Any]:
        self._flush()
        return {
            "contract_version": EXACT_CACHE_CONTRACT_VERSION,
            "path": self.cache_path,
            "sha256": sha256_file(self.cache_path),
            "n_scores": len(self.cache),
        }


def _step_trace(row: Mapping[str, Any]) -> Mapping[str, Any]:
    trace = row["_turn"].get("decision_traces", {}).get("thought")
    if not isinstance(trace, Mapping) or not trace.get("response_ids"):
        raise ValueError("local Thought coordinate has no recorded trace")
    return trace


def _exact_method_scores(
    method: str,
    rows: Sequence[Mapping[str, Any]],
    scorer: ExactTraceScorer,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    local = [row for row in rows if row["level"] in ("step", "thought_token")]
    score_by_step: Dict[Tuple[str, int], Dict[str, Any]] = {}
    row_step: Dict[str, Tuple[str, int]] = {}
    for row in local:
        trace = _step_trace(row)
        step_id = int(row["step_id"])
        trace_key = str(trace.get("trace_sha256", "")) or sha256_json(
            [trace["response_ids"], trace["step_map"]]
        )
        key = (trace_key, step_id)
        row_step[str(row["reference_key"])] = key
        if key not in score_by_step:
            score_by_step[key] = scorer.score(
                trace, step_id, need_progress=(method == "daca_grpo")
            )

    estimates: Dict[str, float] = {}
    diagnostics: Dict[str, Any] = {}
    if method == "daca_grpo":
        grouped: Dict[Tuple[str, str], List[Tuple[str, int]]] = defaultdict(list)
        representative: Dict[Tuple[str, int], Mapping[str, Any]] = {}
        for row in local:
            key = row_step[str(row["reference_key"])]
            representative.setdefault(key, row)
            group = (str(row["environment"]), str(row["game_id"]))
            if key not in grouped[group]:
                grouped[group].append(key)
        z_scores: Dict[Tuple[str, int], float] = {}
        for keys in grouped.values():
            values = [float(score_by_step[key]["progress_delta"]) for key in keys]
            mean = statistics.mean(values)
            scale = statistics.pstdev(values) if len(values) > 1 else 0.0
            for key, value in zip(keys, values):
                z_scores[key] = (value - mean) / (scale + 1e-6) if scale > 0 else 0.0
        for row in local:
            key = row_step[str(row["reference_key"])]
            estimates[str(row["reference_key"])] = float(row["group_advantage"]) * (
                1.0 + 0.1 * z_scores[key]
            )
        diagnostics = {
            "lambda": 0.1,
            "normalization_unit": "environment-game over unique audited Thought steps",
            "n_unique_steps": len(score_by_step),
            "n_terminal_zero_modulation": sum(
                bool(value.get("terminal_step")) for value in score_by_step.values()
            ),
        }
    elif method == "egspo":
        grouped_steps: Dict[Tuple[str, str, int], List[Tuple[str, int]]] = defaultdict(list)
        for row in local:
            key = row_step[str(row["reference_key"])]
            group = (str(row["episode_id"]), str(row["game_id"]), int(row["turn_id"]))
            if key not in grouped_steps[group]:
                grouped_steps[group].append(key)
        selected = set()
        for keys in grouped_steps.values():
            selected.add(
                max(
                    keys,
                    key=lambda key: (
                        float(score_by_step[key]["entropy"]),
                        -int(key[1]),
                        str(key[0]),
                    ),
                )
            )
        for row in local:
            key = row_step[str(row["reference_key"])]
            estimates[str(row["reference_key"])] = (
                float(row["group_advantage"]) if key in selected else 0.0
            )
        diagnostics = {
            "selection": "maximum exact categorical entropy among audited steps in each turn",
            "n_unique_steps": len(score_by_step),
            "n_selected_steps": len(selected),
            "selection_scope": "frozen audited-coordinate set",
        }
    else:
        raise ValueError("exact scorer called for unsupported method")
    return estimates, diagnostics


def build_estimator(
    method: str,
    catalog: Mapping[str, Any],
    model_path: str = "",
    cache_path: str = "",
) -> Dict[str, Any]:
    if method not in METHODS:
        raise ValueError("unsupported V4 credit estimator: {}".format(method))
    rows = list(catalog["rows"])
    predictions: Dict[str, float] = {}
    heads: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}
    exact_receipt = None
    if method == "neda":
        predictions, heads = _neda_oof(rows)
    elif method in ("daca_grpo", "egspo"):
        if not model_path or not cache_path:
            raise ValueError("{} requires model and resumable exact-score cache".format(method))
        scorer = ExactTraceScorer(model_path, cache_path)
        predictions, diagnostics = _exact_method_scores(method, rows, scorer)
        exact_receipt = scorer.finish()
    else:
        for row in rows:
            key = str(row["reference_key"])
            if row["level"] not in METHOD_SUPPORT[method]:
                continue
            if method == "agentic_vrpo":
                predictions[key] = float(row["preference_score"])
            else:
                predictions[key] = float(row["group_advantage"])

    estimates = []
    for row in rows:
        key = str(row["reference_key"])
        supported = row["level"] in METHOD_SUPPORT[method]
        if supported and key not in predictions:
            raise ValueError("{} omitted supported reference {}".format(method, key))
        if not supported and key in predictions:
            raise ValueError("{} scored an N/A coordinate {}".format(method, key))
        value = _finite(predictions[key], "credit estimate") if supported else None
        estimates.append(
            {
                "reference_key": key,
                "environment": row["environment"],
                "task_index": int(row["task_index"]),
                "game_id": row["game_id"],
                "episode_id": row["episode_id"],
                "turn_id": int(row["turn_id"]),
                "level": row["level"],
                "coordinate_id": row["coordinate_id"],
                "step_id": row["step_id"],
                "positions": row["positions"],
                "feature_sha256": row["feature_sha256"],
                "supported": supported,
                "credit": value,
                "group_advantage": float(row["group_advantage"]),
            }
        )
    coverage = {
        level: sum(row["supported"] and row["level"] == level for row in estimates)
        for level in LEVELS
    }
    artifact = {
        "contract_version": ESTIMATOR_CONTRACT_VERSION,
        "status": "PASS",
        "phase": "V4-2",
        "scientific_role": "frozen credit estimate; not an online policy result",
        "method": method,
        "method_semantics": METHOD_SEMANTICS[method],
        "supported_levels": sorted(METHOD_SUPPORT[method]),
        "reference_contract_version": catalog["contract_version"],
        "reference_sha256": catalog["reference_sha256"],
        "source_receipts_sha256": catalog["source_receipts_sha256"],
        "audit_receipts": catalog["audit_receipts"],
        "n_catalog_references": int(catalog["n_references"]),
        "n_supported_estimates": sum(coverage.values()),
        "coverage": coverage,
        "estimates": estimates,
        "heads": heads,
        "method_diagnostics": diagnostics,
        "exact_score_cache": exact_receipt,
        "label_access_contract": (
            "game-group OOF counterfactual targets"
            if method == "neda"
            else "no counterfactual effect read by estimator scoring function"
        ),
        "claim_boundary": (
            "fidelity is computed later from an independent join; trajectory-broadcast controls "
            "do not become local causal estimators merely because they are scoreable"
        ),
    }
    artifact["estimates_sha256"] = sha256_json(
        [[row["reference_key"], row["supported"], row["credit"]] for row in estimates]
    )
    artifact["artifact_sha256"] = sha256_json(artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--alf-audit", required=True)
    parser.add_argument("--web-audit", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--cache", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-small", action="store_true")
    args = parser.parse_args()
    expected = (
        {"alfworld": (0, 0), "webshop": (0, 0)}
        if args.allow_small
        else {"alfworld": (64, 1024), "webshop": (100, 600)}
    )
    catalog = load_reference_catalog(
        {"alfworld": args.alf_audit, "webshop": args.web_audit}, expected
    )
    artifact = build_estimator(
        args.method, catalog, model_path=args.model, cache_path=args.cache
    )
    _atomic_json(artifact, args.out)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "method": artifact["method"],
                "n_catalog_references": artifact["n_catalog_references"],
                "n_supported_estimates": artifact["n_supported_estimates"],
                "coverage": artifact["coverage"],
                "out": os.path.realpath(args.out),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
