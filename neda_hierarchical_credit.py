"""Mass-conserving R/H/D/T credit and game-group OOF utilities for V4-E05."""

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from neda_repro import sha256_json, stable_seed


HIERARCHY_CONTRACT_VERSION = "neda-hierarchical-credit-v1"
OOF_HEAD_CONTRACT_VERSION = "neda-hierarchical-oof-heads-v1"


def zero_sum_l1(values: Sequence[float], epsilon: float = 1e-12) -> List[float]:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("zero-sum normalization requires a non-empty parent")
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    scale = sum(abs(value) for value in centered)
    if scale <= float(epsilon):
        return [0.0] * len(values)
    result = [value / (scale + float(epsilon)) for value in centered]
    # Remove the last few ulps so nested sums remain auditable at tight tolerance.
    result[-1] -= sum(result)
    return result


def normalized_outer_prior(horizon: int, kind: str, gamma: float) -> List[float]:
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("episode horizon must be positive")
    if kind == "uniform":
        return [1.0 / horizon] * horizon
    if kind != "temporal":
        raise ValueError("outer prior must be uniform or temporal")
    if not (0.0 < float(gamma) <= 1.0):
        raise ValueError("gamma must be in (0,1]")
    weights = [float(gamma) ** (horizon - 1 - index) for index in range(horizon)]
    denominator = sum(weights)
    return [value / denominator for value in weights]


def _allocate_parent(
    parent_credit: float, scores: Sequence[float], eta: float
) -> List[float]:
    scores = [float(value) for value in scores]
    if not scores:
        raise ValueError("hierarchy parent has no children")
    if not (0.0 <= float(eta) <= 1.0):
        raise ValueError("eta must be in [0,1]")
    base = float(parent_credit) / len(scores)
    residual = zero_sum_l1(scores)
    values = [base + float(eta) * abs(float(parent_credit)) * value for value in residual]
    values[-1] += float(parent_credit) - sum(values)
    return values


def allocate_hierarchy(
    episode_advantage: float,
    mass_constant: float,
    turn_scores: Sequence[float],
    step_scores: Sequence[Sequence[float]],
    token_scores: Sequence[Sequence[Sequence[float]]],
    eta_h: float,
    eta_d: float,
    eta_t: float,
    outer_prior: str = "uniform",
    gamma: float = 0.95,
) -> Dict[str, Any]:
    """Allocate fixed episode mass through turn, step and token children."""

    turn_scores = [float(value) for value in turn_scores]
    horizon = len(turn_scores)
    if len(step_scores) != horizon or len(token_scores) != horizon:
        raise ValueError("turn/step/token hierarchy shape mismatch")
    mass = float(mass_constant) * float(episode_advantage)
    prior = normalized_outer_prior(horizon, outer_prior, gamma)
    turn_residual = zero_sum_l1(turn_scores)
    turn_credit = [
        mass * prior[index] + float(eta_h) * abs(mass) * turn_residual[index]
        for index in range(horizon)
    ]
    turn_credit[-1] += mass - sum(turn_credit)
    turns = []
    for turn_index in range(horizon):
        steps = [float(value) for value in step_scores[turn_index]]
        if len(token_scores[turn_index]) != len(steps):
            raise ValueError("step/token hierarchy shape mismatch")
        step_credit = _allocate_parent(turn_credit[turn_index], steps, eta_d)
        step_rows = []
        for step_index, value in enumerate(step_credit):
            tokens = [float(x) for x in token_scores[turn_index][step_index]]
            token_credit = _allocate_parent(value, tokens, eta_t)
            step_rows.append(
                {
                    "step_index": step_index,
                    "score": steps[step_index],
                    "credit": value,
                    "tokens": [
                        {"token_index": index, "score": tokens[index], "credit": credit}
                        for index, credit in enumerate(token_credit)
                    ],
                }
            )
        turns.append(
            {
                "turn_index": turn_index,
                "score": turn_scores[turn_index],
                "prior": prior[turn_index],
                "credit": turn_credit[turn_index],
                "steps": step_rows,
            }
        )
    result = {
        "contract_version": HIERARCHY_CONTRACT_VERSION,
        "episode_advantage": float(episode_advantage),
        "mass_constant": float(mass_constant),
        "episode_mass": mass,
        "eta": {"H": float(eta_h), "D": float(eta_d), "T": float(eta_t)},
        "outer_prior": outer_prior,
        "gamma": float(gamma),
        "turns": turns,
    }
    audit_hierarchy(result)
    result["credit_sha256"] = sha256_json(result)
    return result


def audit_hierarchy(artifact: Mapping[str, Any], tolerance: float = 1e-9) -> Dict[str, float]:
    mass = float(artifact["episode_mass"])
    turns = list(artifact["turns"])
    turn_error = abs(sum(float(row["credit"]) for row in turns) - mass)
    step_error = 0.0
    token_error = 0.0
    for turn in turns:
        steps = list(turn["steps"])
        step_error = max(
            step_error,
            abs(sum(float(row["credit"]) for row in steps) - float(turn["credit"])),
        )
        for step in steps:
            token_error = max(
                token_error,
                abs(
                    sum(float(row["credit"]) for row in step["tokens"])
                    - float(step["credit"])
                ),
            )
    if max(turn_error, step_error, token_error) > float(tolerance):
        raise ValueError("hierarchical credit mass conservation failed")
    return {
        "turn_mass_error": turn_error,
        "max_step_parent_error": step_error,
        "max_token_parent_error": token_error,
    }


def grouped_fold_assignment(
    groups: Sequence[str], folds: int, seed: int
) -> Tuple[List[int], Dict[str, int]]:
    folds = int(folds)
    unique = sorted(set(str(value) for value in groups))
    if folds < 2 or len(unique) < folds:
        raise ValueError("OOF requires at least one distinct group per fold")
    ranked = sorted(unique, key=lambda value: (stable_seed(seed, value, "oof"), value))
    mapping = {group: index % folds for index, group in enumerate(ranked)}
    assignments = [mapping[str(value)] for value in groups]
    for group in unique:
        if len({assignments[index] for index, value in enumerate(groups) if str(value) == group}) != 1:
            raise ValueError("one game group crossed OOF folds")
    return assignments, mapping


def grouped_oof_ridge(
    features: Sequence[Sequence[float]],
    targets: Sequence[float],
    groups: Sequence[str],
    folds: int = 4,
    ridge: float = 10.0,
    seed: int = 92001,
) -> Dict[str, Any]:
    """Fit a small OOF ridge head with train-fold-only standardization."""

    x = [[float(value) for value in row] for row in features]
    y = [float(value) for value in targets]
    if (
        not x
        or len(x) != len(y)
        or len(y) != len(groups)
        or not x[0]
        or any(len(row) != len(x[0]) for row in x)
        or any(not math.isfinite(value) for row in x for value in row)
        or any(not math.isfinite(value) for value in y)
    ):
        raise ValueError("invalid OOF feature/target/group shapes")
    assignments, mapping = grouped_fold_assignment(groups, folds, seed)
    predictions = [None] * len(y)
    fold_rows = []
    for fold in range(int(folds)):
        train = [index for index, value in enumerate(assignments) if value != fold]
        test = [index for index, value in enumerate(assignments) if value == fold]
        if len(train) == 0 or len(test) == 0:
            raise ValueError("empty OOF train/test fold")
        width = len(x[0])
        mean = [sum(x[index][j] for index in train) / len(train) for j in range(width)]
        scale = []
        for j in range(width):
            variance = sum(
                (x[index][j] - mean[j]) ** 2 for index in train
            ) / len(train)
            value = math.sqrt(variance)
            scale.append(1.0 if value < 1e-8 else value)
        train_x = [
            [(x[index][j] - mean[j]) / scale[j] for j in range(width)] + [1.0]
            for index in train
        ]
        test_x = [
            [(x[index][j] - mean[j]) / scale[j] for j in range(width)] + [1.0]
            for index in test
        ]
        dimension = width + 1
        matrix = [[0.0] * dimension for _ in range(dimension)]
        vector = [0.0] * dimension
        for row, index in zip(train_x, train):
            for left in range(dimension):
                vector[left] += row[left] * y[index]
                for right in range(dimension):
                    matrix[left][right] += row[left] * row[right]
        for index in range(width):
            matrix[index][index] += float(ridge)
        weights = _solve_linear_system(matrix, vector)
        for index, row in zip(test, test_x):
            predictions[index] = sum(value * weight for value, weight in zip(row, weights))
        train_groups = {str(groups[index]) for index in train}
        test_groups = {str(groups[index]) for index in test}
        if train_groups & test_groups:
            raise ValueError("OOF group leakage detected")
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "train_groups_sha256": sha256_json(sorted(train_groups)),
                "test_groups_sha256": sha256_json(sorted(test_groups)),
                "group_overlap": [],
                # Persist the train-fold-only transform and ridge solution so
                # downstream rows from the held-out games can receive genuine
                # OOF predictions without refitting on their own targets.
                "feature_mean": mean,
                "feature_scale": scale,
                "weights": weights,
            }
        )
    if any(value is None or not math.isfinite(float(value)) for value in predictions):
        raise ValueError("OOF predictions are non-finite")
    predictions = [float(value) for value in predictions]
    residual = sum((target - prediction) ** 2 for target, prediction in zip(y, predictions))
    target_mean = sum(y) / len(y)
    total = sum((target - target_mean) ** 2 for target in y)
    r2 = None if total <= 1e-12 else 1.0 - residual / total
    return {
        "contract_version": OOF_HEAD_CONTRACT_VERSION,
        "predictions": predictions,
        "targets": y,
        "fold_assignments": assignments,
        "fold_map": mapping,
        "fold_rows": fold_rows,
        "folds": int(folds),
        "ridge": float(ridge),
        "seed": int(seed),
        "oof_r2": r2,
        "prediction_sha256": sha256_json(predictions),
    }


def _solve_linear_system(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> List[float]:
    """Solve a small dense system with partial-pivot Gaussian elimination."""

    n = len(vector)
    if n == 0 or len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("invalid linear system shape")
    augmented = [
        [float(value) for value in matrix[index]] + [float(vector[index])]
        for index in range(n)
    ]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("ridge normal equation is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    solution = [augmented[index][-1] for index in range(n)]
    if any(not math.isfinite(value) for value in solution):
        raise ValueError("ridge solution is non-finite")
    return solution


def contract_probes() -> Dict[str, Any]:
    turn = [0.2, -0.1, 0.8]
    steps = [[0.1, 0.9], [0.4, -0.2, 0.3], [0.0, 0.7]]
    tokens = [
        [[0.1, 0.2], [0.4, -0.3, 0.2]],
        [[0.1, 0.9], [0.2, 0.3], [-0.4, 0.5]],
        [[0.0, 0.0], [0.2, 0.6, -0.1]],
    ]
    full = allocate_hierarchy(1.0, 3.0, turn, steps, tokens, 0.5, 0.5, 0.5)
    off = allocate_hierarchy(1.0, 3.0, turn, steps, tokens, 0.0, 0.0, 0.0)
    negative = allocate_hierarchy(-1.0, 3.0, turn, steps, tokens, 1.0, 1.0, 1.0)
    uniform_turn_error = max(
        abs(float(row["credit"]) - 1.0) for row in off["turns"]
    )
    uniform_step_error = max(
        abs(float(step["credit"]) - float(turn_row["credit"]) / len(turn_row["steps"]))
        for turn_row in off["turns"]
        for step in turn_row["steps"]
    )
    uniform_token_error = max(
        abs(float(token["credit"]) - float(step["credit"]) / len(step["tokens"]))
        for turn_row in off["turns"]
        for step in turn_row["steps"]
        for token in step["tokens"]
    )
    maximum = max(uniform_turn_error, uniform_step_error, uniform_token_error)
    if maximum > 1e-9:
        raise ValueError("eta=0 hierarchy did not degenerate to uniform children")
    return {
        "status": "PASS",
        "full_mass_audit": audit_hierarchy(full),
        "negative_mass_audit": audit_hierarchy(negative),
        "eta_zero_uniform_errors": {
            "turn": uniform_turn_error,
            "step": uniform_step_error,
            "token": uniform_token_error,
        },
        "probe_sha256": sha256_json([full, off, negative]),
    }
