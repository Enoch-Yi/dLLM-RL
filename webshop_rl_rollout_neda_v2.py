#!/usr/bin/env python3
"""Isolated WebShop rollout entry point for NeDA-v2.

The frozen WebShop environment, task schedule, Action interface, and rollout
conversion remain unchanged.  This wrapper swaps only the decision function
for the legal-Action contrastive NeDA-v2 decoder and authenticates that
provenance in the shard manifest.
"""

from __future__ import annotations

import json
import os
import sys

import webshop_credit_benchmark as benchmark
import webshop_rl_rollout as rollout
from neda_v2_decision import (
    NEDA_V2_EVIDENCE_CONTRACT_VERSION,
    two_stage_decision_decode,
)


def _output_path(argv):
    for index, value in enumerate(argv):
        if value == "--out" and index + 1 < len(argv):
            return os.path.realpath(argv[index + 1])
        if value.startswith("--out="):
            return os.path.realpath(value.split("=", 1)[1])
    raise ValueError("NeDA-v2 WebShop rollout requires --out")


def _annotate_manifest(path):
    manifest_path = path + ".manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("rl_method") != "neda":
        raise ValueError("NeDA-v2 WebShop shard requires rl_method=neda")
    manifest["rl_variant"] = "neda_v2"
    manifest["credit_evidence_contract"] = (
        NEDA_V2_EVIDENCE_CONTRACT_VERSION
    )
    temporary = manifest_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, manifest_path)


def main():
    benchmark.two_stage_decision_decode = two_stage_decision_decode
    output = _output_path(sys.argv[1:])
    rollout.main()
    _annotate_manifest(output)


if __name__ == "__main__":
    main()
