#!/usr/bin/env python3
"""Fail-closed audit for two-stage Thought/Action boundary artifacts."""

import argparse
import json
import os
import re


ACTION_RE = re.compile(r"Action:\s*(.*)", re.IGNORECASE)
AUDIT_VERSION = "neda-boundary-audit-v1"


def parsed_action(text):
    match = ACTION_RE.search(str(text))
    if match is None:
        return None
    action = match.group(1).split("\n", 1)[0].strip()
    if action.endswith("."):
        action = action[:-1].strip()
    return action or None


def audit_boundary_artifact(artifact):
    rows = []
    if isinstance(artifact, list):
        turn_groups = [(row.get("game_id", "unknown"), [row]) for row in artifact]
        n_games = len({str(row.get("game_id", "unknown")) for row in artifact})
    elif isinstance(artifact, dict):
        results = artifact.get("results", [])
        turn_groups = [
            (game.get("game_id", game_index), game.get("trajectory", []))
            for game_index, game in enumerate(results)
        ]
        n_games = len(results)
    else:
        raise ValueError("boundary artifact must be an eval dict or rollout list")

    for game_id, turns in turn_groups:
        for turn_index, turn in enumerate(turns):
            traces = turn.get("decision_traces", {})
            thought_trace = traces.get("thought", {})
            action_trace = traces.get("action", {})
            boundary = turn.get("decision_boundary") or {}
            generated_action = parsed_action(
                turn.get("generation", turn.get("response", ""))
            )
            raw_action = turn.get("raw_action")
            executed_action = turn.get("executed_action", turn.get("sent_action"))
            rows.append(
                {
                    "game_id": str(game_id),
                    "turn": int(turn.get("step", turn_index)),
                    "generated_action": generated_action,
                    "raw_action": raw_action,
                    "executed_action": executed_action,
                    "generation_raw_match": generated_action == raw_action,
                    "raw_executed_match": raw_action == executed_action,
                    "thought_trace_nonempty": bool(
                        thought_trace.get("response_ids")
                    ),
                    "action_trace_nonempty": bool(action_trace.get("response_ids")),
                    "action_span_nonempty": (
                        list(action_trace.get("action_span", [0, 0]))[1]
                        > list(action_trace.get("action_span", [0, 0]))[0]
                    ),
                    "boundary_overlap_kind": str(
                        boundary.get(
                            "dropped_boundary_overlap_kind", "none"
                        )
                    ),
                    "sent_is_legal": bool(turn.get("sent_is_legal", False)),
                }
            )

    n_turns = len(rows)
    summary = {
        "audit_version": AUDIT_VERSION,
        "n_games": n_games,
        "n_turns": n_turns,
        "generation_raw_mismatch": sum(
            not row["generation_raw_match"] for row in rows
        ),
        "raw_executed_mismatch": sum(not row["raw_executed_match"] for row in rows),
        # Empty Thought is an explicit Action-only policy decision.  It is
        # reported but is not a boundary failure; Action remains the sole
        # environment-facing decision coordinate for that turn.
        "empty_thought_trace": sum(
            not row["thought_trace_nonempty"] for row in rows
        ),
        "action_only_turns": sum(
            not row["thought_trace_nonempty"] and row["action_trace_nonempty"]
            for row in rows
        ),
        "empty_action_trace": sum(not row["action_trace_nonempty"] for row in rows),
        "empty_action_span": sum(not row["action_span_nonempty"] for row in rows),
        "sent_legal_rate": (
            sum(row["sent_is_legal"] for row in rows) / n_turns if n_turns else 0.0
        ),
        "action_marker_prefix_overlap": sum(
            row["boundary_overlap_kind"] == "action-marker-prefix"
            for row in rows
        ),
    }
    summary["status"] = "PASS" if (
        n_turns > 0
        and summary["generation_raw_mismatch"] == 0
        and summary["raw_executed_mismatch"] == 0
        and summary["empty_action_trace"] == 0
        and summary["empty_action_span"] == 0
    ) else "FAIL"
    summary["examples"] = [
        row for row in rows
        if not (
            row["generation_raw_match"]
            and row["raw_executed_match"]
            and row["action_trace_nonempty"]
            and row["action_span_nonempty"]
        )
    ][:20]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    with open(args.artifact, "r", encoding="utf-8") as handle:
        result = audit_boundary_artifact(json.load(handle))
    out = args.out or args.artifact + ".boundary_audit.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
