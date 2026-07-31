#!/usr/bin/env python3
"""Matched U0--U4 horizon-credit estimators for one immutable rollout.

All variants are attached to the same records in the same order. Training picks
``training.advantage_variant``; it must not regenerate rollout data per variant.
"""

import argparse
import copy
import json
import math
import statistics
from collections import defaultdict

from neda_repro import sha256_file, sha256_json


CREDIT_CONTRACT_VERSION = "neda-horizon-credit-v1"
VARIANTS = ("U0", "U1", "U2", "U3", "U4")


def _episode_id(record):
    if "episode_id" in record:
        return str(record["episode_id"])
    if "ep_id" in record:
        return str(record["ep_id"])
    raise ValueError("record has no episode_id/ep_id")


def _turn_id(record):
    if "turn_id" in record:
        return int(record["turn_id"])
    if "h" in record:
        return int(record["h"])
    raise ValueError("record has no turn_id/h")


def _base_advantage(record):
    for key in ("group_advantage", "episode_advantage", "reward"):
        if key in record:
            return float(record[key])
    raise ValueError("record has no group/episode advantage")


def group_episodes(records):
    episodes = defaultdict(list)
    for index, record in enumerate(records):
        episodes[_episode_id(record)].append((index, record))
    ordered = []
    for episode_id, rows in sorted(episodes.items()):
        rows.sort(key=lambda pair: _turn_id(pair[1]))
        turns = [_turn_id(row) for _, row in rows]
        if turns != list(range(len(rows))):
            raise ValueError("episode {} has non-contiguous turns {}".format(episode_id, turns))
        advantages = [_base_advantage(row) for _, row in rows]
        if max(advantages) - min(advantages) > 1e-8:
            raise ValueError(
                "episode {} does not have one shared group advantage".format(episode_id)
            )
        for _, row in rows:
            declared_h = row.get("episode_horizon", row.get("ep_len", len(rows)))
            if int(declared_h) != len(rows):
                raise ValueError(
                    "episode {} horizon mismatch: {} vs {}".format(
                        episode_id, declared_h, len(rows)
                    )
                )
        ordered.append((episode_id, rows, advantages[0]))
    return ordered


def compute_advantage_variants(records, gamma=0.95, mass_constant=None):
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1]")
    episodes = group_episodes(records)
    if not episodes:
        raise ValueError("empty rollout")
    horizons = [len(rows) for _, rows, _ in episodes]
    c_value = float(statistics.mean(horizons) if mass_constant is None else mass_constant)
    if c_value <= 0:
        raise ValueError("mass_constant must be positive")

    u0_values = []
    u1_values = []
    staged = {}
    for episode_id, rows, advantage in episodes:
        horizon = len(rows)
        denom = sum(gamma ** distance for distance in range(horizon))
        values = []
        for position, (index, _) in enumerate(rows):
            distance = horizon - 1 - position
            temporal = gamma ** distance
            u0 = advantage
            u1 = temporal * advantage
            u3 = c_value / horizon * advantage
            u4 = c_value * temporal / denom * advantage
            u0_values.append(u0)
            u1_values.append(u1)
            values.append((index, u0, u1, u3, u4))
        staged[episode_id] = values

    mean_abs_u0 = statistics.mean(abs(value) for value in u0_values)
    mean_abs_u1 = statistics.mean(abs(value) for value in u1_values)
    scale_u2 = mean_abs_u1 / mean_abs_u0 if mean_abs_u0 > 0 else 1.0

    output = [copy.deepcopy(record) for record in records]
    for episode_id, values in staged.items():
        for index, u0, u1, u3, u4 in values:
            output[index]["advantages"] = {
                "U0": u0,
                "U1": u1,
                "U2": scale_u2 * u0,
                "U3": u3,
                "U4": u4,
            }
            output[index]["credit_contract"] = CREDIT_CONTRACT_VERSION
    metadata = {
        "contract_version": CREDIT_CONTRACT_VERSION,
        "gamma": float(gamma),
        "mass_constant": c_value,
        "u2_global_mean_abs_scale": scale_u2,
        "n_episodes": len(episodes),
        "n_turns": len(records),
        "mean_horizon": statistics.mean(horizons),
        "variant_summary": summarize_variants(output),
    }
    metadata["advantages_sha256"] = sha256_json(
        [[row.get("sample_id", i), row["advantages"]] for i, row in enumerate(output)]
    )
    return output, metadata


def summarize_variants(records):
    episodes = group_episodes(records)
    summary = {}
    for variant in VARIANTS:
        values = [float(row["advantages"][variant]) for row in records]
        episode_mass = []
        for _, rows, _ in episodes:
            episode_mass.append(sum(float(row["advantages"][variant]) for _, row in rows))
        summary[variant] = {
            "mean": statistics.mean(values),
            "mean_abs": statistics.mean(abs(value) for value in values),
            "rms": math.sqrt(statistics.mean(value * value for value in values)),
            "min": min(values),
            "max": max(values),
            "mean_abs_episode_mass": statistics.mean(abs(value) for value in episode_mass),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="one immutable raw rollout JSON")
    parser.add_argument("--out", required=True, help="records with all U0--U4 fields")
    parser.add_argument("--summary", help="optional separate metadata JSON")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--mass-constant", type=float)
    parser.add_argument(
        "--select",
        choices=VARIANTS,
        help="also overwrite legacy reward with this variant; all variants remain attached",
    )
    args = parser.parse_args()
    with open(args.data, "r", encoding="utf-8") as handle:
        source = json.load(handle)
    records, metadata = compute_advantage_variants(
        source, gamma=args.gamma, mass_constant=args.mass_constant
    )
    metadata["source_path"] = args.data
    metadata["source_sha256"] = sha256_file(args.data)
    if args.select:
        for record in records:
            record["reward"] = record["advantages"][args.select]
        metadata["selected_variant"] = args.select
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    summary_path = args.summary or args.out + ".credit.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(
        "[credit] wrote {} turns / {} episodes -> {} ({})".format(
            metadata["n_turns"], metadata["n_episodes"], args.out, ",".join(VARIANTS)
        )
    )
    for variant in VARIANTS:
        row = metadata["variant_summary"][variant]
        print(
            "[credit] {} mean_abs={:.6f} episode_mass={:.6f}".format(
                variant, row["mean_abs"], row["mean_abs_episode_mass"]
            )
        )


if __name__ == "__main__":
    main()
