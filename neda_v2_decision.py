#!/usr/bin/env python3
"""NeDA-v2 decision wrapper with legal-Action contrastive evidence.

The frozen NeDA-v1 decoder is reused verbatim for generation and exact replay.
NeDA-v2 changes only the evidence attached to each Thought trace.  Under the
normalized admissible-command Trie, the probability of all legal Actions other
than the executed Action is exactly ``1 - p(a_exec)``.  We can therefore turn
the already recorded teacher-forced path probability into an exact legal-set
log-odds margin without another model forward or an approximation over sampled
negative Actions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

from neda_repro import sha256_json
from neda_v4_decision import (
    two_stage_decision_decode as _v1_two_stage_decision_decode,
)


NEDA_V2_EVIDENCE_CONTRACT_VERSION = "neda-v2-legal-action-margin-v1"
_LOG_PROBABILITY_CEILING = -1.0e-12


def _log1mexp(log_probability: float) -> float:
    """Return log(1-exp(x)) stably for x <= 0.

    Exact float32 scoring can round an overwhelmingly likely Trie leaf to
    log-probability zero.  Clipping by 1e-12 keeps the corresponding log odds
    finite while preserving the intended ordering.
    """

    value = float(log_probability)
    if not math.isfinite(value):
        raise ValueError("legal Action path log-probability is non-finite")
    if value > 1.0e-6:
        raise ValueError("legal Action path log-probability exceeds zero")
    value = min(value, _LOG_PROBABILITY_CEILING)
    if value < -math.log(2.0):
        return math.log1p(-math.exp(value))
    return math.log(-math.expm1(value))


def build_legal_action_margin_evidence(
    executed_evidence: Mapping[str, Any],
    action_trace: Mapping[str, Any],
) -> Dict[str, Any]:
    """Convert NeDA-v1 executed-path scores into NeDA-v2 log-odds margins."""

    if executed_evidence.get("contract_version") != "neda-action-evidence-v1":
        raise ValueError("NeDA-v2 requires frozen NeDA-v1 Action evidence")
    if executed_evidence.get("allocation_status") == "ACTION_ONLY":
        result = {
            "contract_version": NEDA_V2_EVIDENCE_CONTRACT_VERSION,
            "estimator": "not applicable: Action-only decision",
            "interpretation": (
                "No synthetic Thought credit; optimize only the recorded "
                "environment-facing Action coordinate"
            ),
            "allocation_status": "ACTION_ONLY",
            "legal_support": "exact admissible-command Trie leaves",
            "boundaries": [],
            "executed_action_logprobs": [],
            "other_legal_action_logprobs": [],
            "legal_action_log_odds": [],
            "segments": [],
            "source_evidence_sha256": sha256_json(dict(executed_evidence)),
        }
        result["evidence_sha256"] = sha256_json(result)
        return result

    token_rows = executed_evidence.get("token_logprobs")
    segments = executed_evidence.get("segments")
    boundaries = [int(value) for value in executed_evidence.get("boundaries", [])]
    if (
        not isinstance(token_rows, list)
        or not isinstance(segments, list)
        or len(token_rows) != len(boundaries) + 1
        or len(segments) != len(boundaries)
    ):
        raise ValueError("NeDA-v1 evidence boundary geometry is invalid")

    action_ids = [int(value) for value in action_trace.get("response_ids", [])]
    allowed_rows = action_trace.get("constraint_allowed_token_ids", [])
    if not action_ids or len(allowed_rows) != len(action_ids):
        raise ValueError("NeDA-v2 requires the exact executed Action Trie support")
    has_legal_alternative = any(
        len({int(token) for token in allowed}) > 1 for allowed in allowed_rows
    )

    path_logprobs = []
    other_logprobs = []
    margins = []
    for row in token_rows:
        values = [float(value) for value in row]
        if len(values) != len(action_ids):
            raise ValueError("Action evidence token count drift")
        log_probability = math.fsum(values)
        if log_probability > 1.0e-5:
            raise ValueError("executed Action probability exceeds one")
        path_logprobs.append(log_probability)
        if has_legal_alternative:
            log_other = _log1mexp(log_probability)
            other_logprobs.append(log_other)
            margins.append(log_probability - log_other)
        else:
            # With a singleton legal support there is no identifiable
            # contrast.  A neutral constant makes every segment delta zero.
            other_logprobs.append(None)
            margins.append(0.0)

    v2_segments = []
    for index, source in enumerate(segments):
        before = float(margins[index])
        after = float(margins[index + 1])
        v2_segments.append(
            {
                "segment_id": int(source["segment_id"]),
                "left_round_exclusive": int(source["left_round_exclusive"]),
                "right_round_inclusive": int(source["right_round_inclusive"]),
                "member_rounds": [
                    int(value) for value in source["member_rounds"]
                ],
                "legal_action_margin_before": before,
                "legal_action_margin_after": after,
                "legal_action_margin_delta": after - before,
            }
        )

    result = {
        "contract_version": NEDA_V2_EVIDENCE_CONTRACT_VERSION,
        "estimator": (
            "exact executed-vs-other-legal Action log-odds under the frozen "
            "admissible-command Trie"
        ),
        "interpretation": (
            "StepMerge allocation evidence, not ground-truth causal credit"
        ),
        "allocation_status": (
            "CONTRASTIVE" if has_legal_alternative else "NO_LEGAL_ALTERNATIVE"
        ),
        "legal_support": "exact admissible-command Trie terminal leaves",
        "other_mass_identity": "p(other legal Actions)=1-p(executed Action)",
        "boundaries": boundaries,
        "executed_action_logprobs": path_logprobs,
        "other_legal_action_logprobs": other_logprobs,
        "legal_action_log_odds": margins,
        "segments": v2_segments,
        "source_evidence_sha256": sha256_json(dict(executed_evidence)),
    }
    result["evidence_sha256"] = sha256_json(result)
    return result


def two_stage_decision_decode(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Run the frozen v1 decoder and attach the isolated NeDA-v2 evidence."""

    result = _v1_two_stage_decision_decode(*args, **kwargs)
    if result.get("rl_method") != "neda":
        raise ValueError("NeDA-v2 wrapper requires rl_method=neda")
    thought = result["decision_traces"]["thought"]
    action = result["decision_traces"]["action"]
    thought["action_evidence_v2"] = build_legal_action_margin_evidence(
        thought.get("action_evidence", {}), action
    )
    trace_body = dict(thought)
    trace_body.pop("trace_sha256", None)
    thought["trace_sha256"] = sha256_json(trace_body)
    result["rl_variant"] = "neda_v2"
    result["credit_evidence_contract"] = NEDA_V2_EVIDENCE_CONTRACT_VERSION
    return result
