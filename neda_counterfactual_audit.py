#!/usr/bin/env python3
"""Fail-closed audit for NeDA V4-E04/O01 counterfactual artifacts."""

import argparse
import json
import math
import os
from typing import Any, Dict, List, Mapping, Sequence

from neda_counterfactual import (
    COUNTERFACTUAL_CONTRACT_VERSION,
    MEMORY_CONTRACT,
    summarize_paired_effects,
    summarize_test_retest,
)
from neda_repro import sha256_file


AUDIT_CONTRACT_VERSION = "neda-counterfactual-audit-v1"


def _normal_ci_is_useful(reference: Mapping[str, Any]) -> bool:
    """Pre-registered pilot heuristic for a bounded return difference in [-1,1]."""

    interval = reference.get("normal_95ci")
    if not isinstance(interval, list) or len(interval) != 2:
        return False
    lower, upper = float(interval[0]), float(interval[1])
    # An interval covering the entire feasible effect range is not informative.
    return not (lower <= -1.0 and upper >= 1.0)


def merge_counterfactual_artifacts(artifacts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not artifacts:
        raise ValueError("at least one counterfactual artifact is required")
    for artifact in artifacts:
        if artifact.get("contract_version") != COUNTERFACTUAL_CONTRACT_VERSION:
            raise ValueError("unsupported counterfactual artifact contract")
        if not artifact.get("complete"):
            raise ValueError("cannot merge an incomplete counterfactual shard")
    base_hashes = {str(row.get("base_artifact_sha256")) for row in artifacts}
    model_ids = {str(row.get("model_identity_sha256")) for row in artifacts}
    phases = {str(row.get("phase")) for row in artifacts}
    if len(base_hashes) != 1:
        raise ValueError("branch shards do not share one base artifact SHA")
    if len(model_ids) != 1:
        raise ValueError("branch shards do not share one behavior checkpoint")
    if len(phases) != 1:
        raise ValueError("branch shards do not share one registered phase")
    anchors: List[Dict[str, Any]] = []
    checks: List[Dict[str, Any]] = []
    retest_anchor_ids = set()
    seen = set()
    for artifact in artifacts:
        retest_anchor_ids.update(
            str(value) for value in artifact.get("protocol", {}).get("retest_anchor_ids", [])
        )
        for anchor in artifact.get("anchors", []):
            anchor_id = str(anchor.get("anchor_id"))
            if anchor_id in seen:
                raise ValueError("duplicate anchor across branch shards: {}".format(anchor_id))
            seen.add(anchor_id)
            anchors.append(dict(anchor))
        checks.extend(dict(row) for row in artifact.get("reproducibility_checks", []))
    first = artifacts[0]
    protocol = dict(first.get("protocol", {}))
    for artifact in artifacts[1:]:
        candidate = dict(artifact.get("protocol", {}))
        for key in (
            "k",
            "branch_seed",
            "selection_seed",
            "retest_fraction",
            "intervention_policy",
            "policy_mixture_probability",
            "max_steps",
            "max_history",
            "global_retest_anchor_ids_sha256",
        ):
            if candidate.get(key) != protocol.get(key):
                raise ValueError("branch shard protocol mismatch for {}".format(key))
    protocol["retest_anchor_ids"] = sorted(retest_anchor_ids)
    return {
        "contract_version": COUNTERFACTUAL_CONTRACT_VERSION,
        "artifact_kind": "merged-branch-results",
        "phase": first.get("phase"),
        "memory_contract": first.get("memory_contract"),
        "base_artifact_sha256": next(iter(base_hashes)),
        "model_identity_sha256": next(iter(model_ids)),
        "protocol": protocol,
        "anchors": sorted(anchors, key=lambda row: str(row["anchor_id"])),
        "reproducibility_checks": checks,
        "source_shards": len(artifacts),
        "complete": True,
    }


def audit_counterfactual_artifact(
    artifact: Mapping[str, Any], phase: str = "auto"
) -> Dict[str, Any]:
    if artifact.get("contract_version") != COUNTERFACTUAL_CONTRACT_VERSION:
        raise ValueError("unsupported counterfactual artifact contract")
    if artifact.get("memory_contract") != MEMORY_CONTRACT:
        raise ValueError("counterfactual artifact changed the registered agent memory contract")
    if artifact.get("artifact_kind") not in ("branch-results", "merged-branch-results"):
        raise ValueError("audit requires branch results, not a base-only artifact")
    if not artifact.get("complete"):
        raise ValueError("counterfactual branch artifact is incomplete")

    protocol = artifact.get("protocol", {})
    expected_k = int(protocol.get("k", 0))
    if expected_k < 1:
        raise ValueError("protocol requires K>=1")
    selected_retest = set(str(value) for value in protocol.get("retest_anchor_ids", []))
    anchors = list(artifact.get("anchors", []))
    if not anchors:
        raise ValueError("branch artifact has no anchors")

    prefix_failures: List[str] = []
    crn_failures: List[str] = []
    count_failures: List[str] = []
    reference_failures: List[str] = []
    reference_rows: List[Dict[str, Any]] = []
    useful_se = 0
    total_references = 0

    for anchor in anchors:
        anchor_id = str(anchor.get("anchor_id"))
        repeats = anchor.get("repeats", {})
        expected_repeat_ids = ["0"] + (["1"] if anchor_id in selected_retest else [])
        if sorted(repeats.keys()) != expected_repeat_ids:
            count_failures.append(
                "{} repeats={} expected={}".format(
                    anchor_id, sorted(repeats.keys()), expected_repeat_ids
                )
            )
        repeat_references: Dict[str, Any] = {}
        for repeat_id in expected_repeat_ids:
            if repeat_id not in repeats:
                continue
            pairs = list(repeats[repeat_id].get("pairs", []))
            if len(pairs) != expected_k:
                count_failures.append(
                    "{} repeat {} has {} pairs, expected {}".format(
                        anchor_id, repeat_id, len(pairs), expected_k
                    )
                )
            for pair in pairs:
                for role in ("original", "alternative"):
                    branch = pair.get(role, {})
                    prefix = branch.get("prefix_replay", {})
                    schedule = pair.get("crn_schedule", {})
                    prefix_valid = (
                        prefix.get("status") == "PASS"
                        and prefix.get("env_reset_seed") == schedule.get("env_reset_seed")
                        and prefix.get("anchor_state_sha256") == anchor.get("state_sha256")
                    )
                    if role == "original":
                        prefix_valid = prefix_valid and (
                            branch.get("original_anchor_transition_reproduced") is True
                        )
                    if not prefix_valid:
                        prefix_failures.append(
                            "{} repeat {} sample {} {}".format(
                                anchor_id, repeat_id, pair.get("sample_id"), role
                            )
                        )
                    if branch.get("crn_schedule_sha256") != schedule.get("schedule_sha256"):
                        crn_failures.append(
                            "{} repeat {} sample {} {}".format(
                                anchor_id, repeat_id, pair.get("sample_id"), role
                            )
                        )
                    continuation = list(schedule.get("continuation_decision_seeds", []))
                    for turn in branch.get("trajectory", [])[1:]:
                        offset = int(turn.get("turn_offset", -1)) - 1
                        if (
                            offset < 0
                            or offset >= len(continuation)
                            or int(turn.get("decision_seed", -1)) != int(continuation[offset])
                        ):
                            crn_failures.append(
                                "{} repeat {} sample {} {} continuation-offset-{}".format(
                                    anchor_id,
                                    repeat_id,
                                    pair.get("sample_id"),
                                    role,
                                    offset,
                                )
                            )
            try:
                reference = summarize_paired_effects(pairs)
            except (KeyError, TypeError, ValueError) as error:
                reference_failures.append(
                    "{} repeat {}: {}".format(anchor_id, repeat_id, error)
                )
                continue
            stored = repeats[repeat_id].get("reference")
            if stored is not None and stored != reference:
                reference_failures.append(
                    "{} repeat {} stored reference drift".format(anchor_id, repeat_id)
                )
            repeat_references[repeat_id] = reference
            total_references += 1
            useful_se += int(_normal_ci_is_useful(reference))
        reference_rows.append(
            {"anchor_id": anchor_id, "repeat_references": repeat_references}
        )

    checks = list(artifact.get("reproducibility_checks", []))
    reproducibility_failures = [
        str(row.get("check_id", row.get("anchor_id", "unknown")))
        for row in checks
        if not bool(row.get("pass", False))
    ]
    reliability = summarize_test_retest(reference_rows)
    useful_fraction = useful_se / float(total_references) if total_references else 0.0

    resolved_phase = str(artifact.get("phase")) if phase == "auto" else str(phase)
    e04_pass = not (
        prefix_failures
        or crn_failures
        or count_failures
        or reference_failures
        or reproducibility_failures
    ) and bool(checks)
    if resolved_phase == "e04_smoke":
        gate_pass = e04_pass
        gate_reason = "E04 prefix/CRN/duplicate replay gate"
    elif resolved_phase == "o01":
        spearman = reliability.get("spearman")
        gate_pass = (
            e04_pass
            and reliability.get("n_anchors", 0) == len(selected_retest)
            and spearman is not None
            and float(spearman) >= 0.5
            and useful_fraction > 0.5
        )
        gate_reason = "O01 requires E04 PASS, retest Spearman>=0.5, and >50% useful MC CIs"
    else:
        raise ValueError("phase must resolve to e04_smoke or o01")

    return {
        "contract_version": AUDIT_CONTRACT_VERSION,
        "status": "PASS" if gate_pass else "FAIL",
        "phase": resolved_phase,
        "gate_reason": gate_reason,
        "n_anchors": len(anchors),
        "expected_k": expected_k,
        "n_retest_selected": len(selected_retest),
        "n_references": total_references,
        "state_prefix_replay_pass": not prefix_failures,
        "crn_pairing_pass": not crn_failures,
        "duplicate_branch_reproducibility_pass": bool(checks) and not reproducibility_failures,
        "reference_count_pass": not count_failures,
        "reference_recompute_pass": not reference_failures,
        "useful_reference_ci_fraction": useful_fraction,
        "test_retest": reliability,
        "failures": {
            "prefix_replay": prefix_failures,
            "crn_pairing": crn_failures,
            "counts": count_failures,
            "reference": reference_failures,
            "reproducibility": reproducibility_failures,
        },
    }


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", nargs="+", required=True)
    parser.add_argument("--phase", choices=("auto", "e04_smoke", "o01"), default="auto")
    parser.add_argument("--out")
    args = parser.parse_args()
    loaded = [_load(path) for path in args.artifacts]
    merged = loaded[0] if len(loaded) == 1 else merge_counterfactual_artifacts(loaded)
    audit = audit_counterfactual_artifact(merged, phase=args.phase)
    audit["artifact_paths"] = [os.path.realpath(path) for path in args.artifacts]
    audit["artifact_sha256"] = [sha256_file(path) for path in args.artifacts]
    text = json.dumps(audit, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
