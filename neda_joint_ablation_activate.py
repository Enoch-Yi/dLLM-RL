#!/usr/bin/env python3
"""Issue the fail-closed activation receipt for registered NeDA ablations.

The receipt cannot be created from a training log or an unaudited checkpoint.
It requires:

* the frozen ablation registry and its sidecar;
* an authenticated 8-GPU, seed-12001 NeDA single-run formal audit;
* an authenticated 4-GPU ZeRO-2 exact-replay-only receipt; and
* a scheduler snapshot proving that no ``njab_*`` ablation job was already
  submitted when the prospective activation decision was recorded.

This tool never submits a job and never creates a scientific result.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
from typing import Any, Dict, Mapping, Tuple


CONTRACT_VERSION = "neda-joint-ablation-activation-v1"
MAIN_AUDIT_CONTRACT = "neda-joint-single-formal-run-audit-v1"
REPLAY_RECEIPT_CONTRACT = "neda-v4-multitrace-replay-probe-v1"
REPLAY_DIAGNOSTIC_CONTRACT = "neda-v4-replay-drift-diagnostic-v1"
ABLATION_REGISTRY_CONTRACT = "neda-joint-alfworld-ablations-v1"
ABLATION_JOB_PREFIX = "njab_"
REPLAY_TOLERANCE = 0.05


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(os.path.realpath(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: str) -> Any:
    with open(os.path.realpath(path), encoding="utf-8") as handle:
        return json.load(handle)


def verify_embedded(
    value: Mapping[str, Any], field: str, label: str
) -> None:
    body = dict(value)
    declared = body.pop(field, None)
    if not declared or sha256_json(body) != str(declared):
        raise ValueError("{} embedded SHA drift".format(label))


def atomic_json(value: Mapping[str, Any], path: str) -> None:
    path = os.path.realpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(value: str, path: str) -> None:
    path = os.path.realpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(value)
        if value and not value.endswith("\n"):
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_sidecar(registry_path: str, sidecar_path: str) -> str:
    registry_path = os.path.realpath(registry_path)
    sidecar_path = os.path.realpath(sidecar_path)
    actual = sha256_file(registry_path)
    with open(sidecar_path, encoding="utf-8") as handle:
        fields = handle.read().strip().split()
    if (
        len(fields) != 2
        or fields[0] != actual
        or fields[1] != os.path.basename(registry_path)
    ):
        raise ValueError("ablation registry sidecar drift")
    return actual


def _validate_main_audit(path: str) -> Dict[str, Any]:
    audit = load_json(path)
    verify_embedded(audit, "artifact_sha256", "main formal audit")
    training = audit.get("training") or {}
    if (
        audit.get("contract_version") != MAIN_AUDIT_CONTRACT
        or audit.get("status") != "PASS"
        or audit.get("scientific_result") is not True
        or audit.get("aggregation_complete") is not False
        or audit.get("environment") != "alfworld"
        or audit.get("method") != "neda"
        or int(audit.get("train_seed", -1)) != 12001
        or int(audit.get("world_size", -1)) != 8
        or audit.get("main_table_eligible_topology") is not True
        or int(audit.get("n_frozen_tasks", -1)) != 134
        or int(training.get("environment_interactions", 0)) <= 0
        or int(training.get("signal_groups", 0)) <= 0
        or int(training.get("nonzero_episodes", 0)) <= 0
        or int(training.get("optimizer_steps", 0)) <= 0
    ):
        raise ValueError(
            "activation requires the audited seed-12001 8-GPU NeDA run"
        )
    provenance = audit.get("provenance") or {}
    for path_key, sha_key in (
        ("run_complete", "run_complete_sha256"),
        ("evaluation", "evaluation_sha256"),
    ):
        reference = str(provenance.get(path_key, ""))
        if (
            not os.path.isfile(reference)
            or sha256_file(reference) != str(provenance.get(sha_key, ""))
        ):
            raise ValueError(
                "main formal audit provenance drift: {}".format(path_key)
            )
    iterations = list(provenance.get("iterations") or [])
    if len(iterations) != 3 or any(
        int(value.get("optimizer_steps", 0)) <= 0
        or int(value.get("n_signal_groups", 0)) <= 0
        or int(value.get("n_nonzero_episodes", 0)) <= 0
        for value in iterations
    ):
        raise ValueError("main formal audit lacks three non-degenerate updates")
    return audit


def _validate_replay_receipt(path: str) -> Dict[str, Any]:
    receipt = load_json(path)
    verify_embedded(receipt, "receipt_sha256", "multi-GPU replay receipt")
    padding = receipt.get("distributed_padding") or {}
    drift = receipt.get("rollout_replay_drift") or {}
    if (
        receipt.get("contract_version") != REPLAY_RECEIPT_CONTRACT
        or receipt.get("status") != "PASS"
        or receipt.get("registered_method") != "neda"
        or int(padding.get("world_size", -1)) != 4
        or padding.get("zero_credit") is not True
        or int(padding.get("implicit_accelerate_repeats", -1)) != 0
        or int(receipt.get("unpadded_native_replay_rows", 0)) <= 0
        or int(receipt.get("n_native_replay_rows", 0))
        < int(receipt.get("unpadded_native_replay_rows", 0))
    ):
        raise ValueError("4-GPU ZeRO-2 replay receipt contract drift")
    for source in ("thought", "action", "position"):
        summary = drift.get(source) or {}
        if (
            int(summary.get("n", 0)) <= 0
            or float(summary.get("max_abs", float("inf")))
            > REPLAY_TOLERANCE
        ):
            raise ValueError(
                "4-GPU replay tolerance failed for {}".format(source)
            )
    diagnostic_path = str(receipt.get("replay_diagnostic", ""))
    if (
        not os.path.isfile(diagnostic_path)
        or sha256_file(diagnostic_path)
        != str(receipt.get("replay_diagnostic_sha256", ""))
    ):
        raise ValueError("4-GPU replay diagnostic lineage drift")
    diagnostic = load_json(diagnostic_path)
    verify_embedded(diagnostic, "receipt_sha256", "replay diagnostic")
    parameter_audit = diagnostic.get("parameter_audit") or {}
    if (
        diagnostic.get("contract_version") != REPLAY_DIAGNOSTIC_CONTRACT
        or diagnostic.get("status") != "PASS"
        or diagnostic.get("registered_method") != "neda"
        or int(diagnostic.get("world_size", -1)) != 4
        or list(diagnostic.get("failed_sources") or [])
        or parameter_audit.get("full_parameter_update") is not True
        or int(parameter_audit.get("policy_parameters", -1))
        != int(parameter_audit.get("policy_trainable_parameters", -2))
    ):
        raise ValueError("4-GPU replay diagnostic contract drift")
    for key in ("thought", "action"):
        data_path = str(diagnostic.get(key + "_data", ""))
        if (
            not os.path.isfile(data_path)
            or sha256_file(data_path)
            != str(diagnostic.get(key + "_data_sha256", ""))
        ):
            raise ValueError(
                "4-GPU replay source lineage drift: {}".format(key)
            )
    return receipt


def _parse_ablation_jobs(queue_text: str) -> Tuple[str, ...]:
    jobs = []
    for line in str(queue_text).splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0].isdigit():
            if fields[2].startswith(ABLATION_JOB_PREFIX):
                jobs.append(fields[0])
    return tuple(sorted(set(jobs)))


def build_activation(
    *,
    registry_path: str,
    registry_sidecar_path: str,
    main_audit_path: str,
    replay_receipt_path: str,
    queue_text: str,
    queue_snapshot_path: str,
    activated_at_utc: str,
) -> Dict[str, Any]:
    registry_path = os.path.realpath(registry_path)
    registry = load_json(registry_path)
    registry_sha = _validate_sidecar(
        registry_path, registry_sidecar_path
    )
    if (
        registry.get("contract_version") != ABLATION_REGISTRY_CONTRACT
        or registry.get("status")
        not in (
            "REGISTERED_OFFLINE_MATERIALIZATION_PASS_NOT_ACTIVE",
            "PREPARED_NOT_ACTIVE",
        )
        or registry.get("registered_before_main_seed12001_result") is not True
    ):
        raise ValueError("ablation registry is not prospectively frozen")
    _validate_main_audit(main_audit_path)
    _validate_replay_receipt(replay_receipt_path)
    ablation_jobs = _parse_ablation_jobs(queue_text)
    if ablation_jobs:
        raise ValueError(
            "ablation jobs predate activation: {}".format(
                ",".join(ablation_jobs)
            )
        )
    atomic_text(queue_text, queue_snapshot_path)
    result: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "status": "PASS",
        "scientific_result": False,
        "activated_at_utc": str(activated_at_utc),
        "main_reward_protocol_fixed": True,
        "seed12001_main_non_degenerate_update": True,
        "multi_gpu_exact_replay_gate_passed": True,
        "activation_before_ablation_submission": True,
        "ablation_registry": os.path.realpath(registry_path),
        "ablation_registry_sha256": registry_sha,
        "main_run_audit": {
            "path": os.path.realpath(main_audit_path),
            "sha256": sha256_file(main_audit_path),
        },
        "multi_gpu_replay_audit": {
            "path": os.path.realpath(replay_receipt_path),
            "sha256": sha256_file(replay_receipt_path),
        },
        "queue_snapshot": {
            "path": os.path.realpath(queue_snapshot_path),
            "sha256": sha256_file(queue_snapshot_path),
            "ablation_job_prefix": ABLATION_JOB_PREFIX,
            "matching_job_ids": list(ablation_jobs),
        },
        "claim_boundary": (
            "This receipt only activates registered training jobs. It is not "
            "an ablation result and contains no performance estimate."
        ),
    }
    result["artifact_sha256"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--registry-sidecar", required=True)
    parser.add_argument("--main-run-audit", required=True)
    parser.add_argument("--multi-gpu-replay-receipt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    queue = subprocess.run(
        ["qstat"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    ).stdout
    snapshot = os.path.realpath(args.out) + ".qstat.txt"
    activated_at = (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    result = build_activation(
        registry_path=args.registry,
        registry_sidecar_path=args.registry_sidecar,
        main_audit_path=args.main_run_audit,
        replay_receipt_path=args.multi_gpu_replay_receipt,
        queue_text=queue,
        queue_snapshot_path=snapshot,
        activated_at_utc=activated_at,
    )
    if os.path.exists(args.out):
        existing = load_json(args.out)
        if existing != result:
            raise ValueError("refusing to overwrite a different activation receipt")
    else:
        atomic_json(result, args.out)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
