import copy
import json
import math
import os
import tempfile
import unittest

from neda_hierarchical_credit_builder import (
    ACTION_TOKEN_FEATURES,
    STEP_FEATURES,
    THOUGHT_TOKEN_FEATURES,
    TURN_FEATURES,
)
from neda_data_contract import (
    NATIVE_THOUGHT_REPLAY_CONTRACT_VERSION,
    NATIVE_THOUGHT_SCORING_LAYOUT,
    validate_decision_trace,
)
from neda_v4_multitrace import (
    distributed_step_plan,
    pad_native_rows_for_distributed,
    validate_action_replay_support,
)
from neda_repro import sha256_json
from neda_v4_online_credit import (
    _episodes,
    _project_alfworld_estimator,
    materialize,
)
from neda_v4_multitrace import build_native_rows


def sampling(constraint="none"):
    return {
        "contract_version": "neda-sampling-v1",
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "constraint": constraint,
        "logprob_space": "post_transform",
        "logprob_dtype": "float32",
    }


def trace(kind, ids, steps):
    value = {
        "kind": kind,
        "prefix_ids": [10, 11],
        "response_ids": list(ids),
        "step_map": list(steps),
        "behavior_logprobs": [-0.2 - 0.1 * index for index in range(len(ids))],
        "commit_confidence": [0.8 - 0.1 * index for index in range(len(ids))],
        "thought_span": [0, len(ids)] if kind == "thought" else [0, 0],
        "action_span": [0, 0] if kind == "thought" else [0, len(ids)],
        "sampling": sampling(),
    }
    if kind == "action":
        value["replay_width"] = len(ids)
        value["scoring_layout"] = "full-duplicate-ar-v1"
    else:
        # Prefix length is two and AO block size is four, so the behavior
        # generator completes six response positions.  Only the first three
        # belong to the interface-visible Thought objective.
        replay_ids = list(ids) + [90, 91, 92]
        replay_steps = list(steps) + [2, 2, 3]
        replay_behavior = value["behavior_logprobs"] + [-0.6, -0.7, -0.8]
        replay_confidence = value["commit_confidence"] + [0.5, 0.4, 0.3]
        native = {
            "contract_version": NATIVE_THOUGHT_REPLAY_CONTRACT_VERSION,
            "scoring_layout": NATIVE_THOUGHT_SCORING_LAYOUT,
            "block_size": 4,
            "replay_width": 10,
            "response_ids": replay_ids,
            "step_map": replay_steps,
            "behavior_logprobs": replay_behavior,
            "commit_confidence": replay_confidence,
            "optimization_mask": [True, True, True, False, False, False],
            "position_policy": "mapg_logit",
            "position_trace": [],
        }
        remaining = list(range(len(replay_ids)))
        planned_rounds = sorted(set(replay_steps))
        for round_id in planned_rounds:
            selected = [
                index
                for index, value in enumerate(replay_steps)
                if int(value) == int(round_id)
            ]
            native["position_trace"].append(
                {
                    "contract_version": "neda-position-commitment-v1",
                    "round_id": int(round_id),
                    "policy": "mapg_logit",
                    "temperature": 0.5,
                    "candidate_positions": list(remaining),
                    "selected_positions": selected,
                    "behavior_logprob": 0.0,
                    "current_block_positions": list(range(len(replay_ids))),
                    "timestep": float(round_id) / max(len(planned_rounds), 1),
                }
            )
            remaining = [
                value for value in remaining if value not in set(selected)
            ]
        native["replay_sha256"] = sha256_json(native)
        value["native_replay"] = native
    value["trace_sha256"] = sha256_json(value)
    return value


def row(episode_id, rollout_id, episode_return, advantage):
    return {
        "prompt": "p",
        "response": "r",
        "reward": advantage,
        "sample_id": episode_id + "-t0",
        "game_id": "game-1",
        "group_id": "group-1",
        "episode_id": episode_id,
        "rollout_id": rollout_id,
        "turn_id": 0,
        "episode_horizon": 1,
        "episode_return": episode_return,
        "group_advantage": advantage,
        "turn_reward": episode_return,
        "model_identity_sha256": "a" * 64,
        "raw_action": "look",
        "executed_action": "look",
        "state_before": {
            "contract_version": "neda-online-credit-state-v1",
            "admissible_commands": ["look", "inventory"],
            "cumulative_reward": 0.0,
        },
        "decision_traces": {
            "thought": trace("thought", [21, 22, 23], [0, 0, 1]),
            "action": trace("action", [31, 32], [0, 1]),
        },
    }


def zero_head(level, names):
    width = len(names)
    return {
        "environment": "alfworld",
        "level": level,
        "feature_names": list(names),
        "n_features": width,
        "deployment_model": {
            "feature_mean": [0.0] * width,
            "feature_scale": [1.0] * width,
            "weights": [0.0] * (width + 1),
        },
    }


def estimator():
    result = {
        "heads": {
            "alfworld:turn": zero_head("turn", TURN_FEATURES),
            "alfworld:step": zero_head("step", STEP_FEATURES),
            "alfworld:thought_token": zero_head(
                "thought_token", THOUGHT_TOKEN_FEATURES
            ),
            "alfworld:action_token": zero_head(
                "action_token", ACTION_TOKEN_FEATURES
            ),
        },
        "artifact_sha256": "b" * 64,
    }
    return result


def full_estimator():
    result = estimator()
    for head in result["heads"].values():
        head["head_sha256"] = "a" * 64
    for level, names in (
        ("turn", TURN_FEATURES),
        ("step", STEP_FEATURES),
        ("thought_token", THOUGHT_TOKEN_FEATURES),
        ("action_token", ACTION_TOKEN_FEATURES),
    ):
        head = zero_head(level, names)
        head["environment"] = "webshop"
        head["head_sha256"] = "b" * 64
        result["heads"]["webshop:" + level] = head
    return result


class OnlineCreditTest(unittest.TestCase):
    def setUp(self):
        # For returns [1,0], pstdev=.5 and the implementation's epsilon gives
        # exactly +/- .5/(.5+1e-6).
        self.advantage = 0.5 / 0.500001
        self.rows = [
            row("episode-pos", 0, 1.0, self.advantage),
            row("episode-neg", 1, 0.0, -self.advantage),
        ]

    def test_full_credit_covers_both_recorded_policy_traces(self):
        artifact, thought, action = materialize(
            self.rows, estimator(), 0.5, 0.5, 0.5, "uniform", 0.95, 1.0
        )
        self.assertEqual(artifact["status"], "PASS")
        self.assertEqual(len(thought), 2)
        self.assertEqual(len(action), 2)
        self.assertEqual(len(thought[0]["adv_map"]), 3)
        self.assertEqual(len(action[0]["adv_map"]), 2)
        for thought_row, action_row in zip(thought, action):
            total = sum(thought_row["adv_map"]) + sum(action_row["adv_map"])
            self.assertAlmostEqual(total, thought_row["group_advantage"], places=9)
            self.assertEqual(thought_row["credit_trace_kind"], "thought")
            self.assertEqual(action_row["credit_trace_kind"], "action")

    def test_full_eight_head_artifact_projects_only_alfworld(self):
        projected = _project_alfworld_estimator(full_estimator())
        self.assertEqual(
            set(projected["heads"]),
            {"alfworld:" + level for level in (
                "turn", "step", "thought_token", "action_token"
            )},
        )
        self.assertEqual(projected["deployment_head_environment"], "alfworld")

    def test_truncated_four_head_artifact_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "environment/head set drift"):
            _project_alfworld_estimator(estimator())

    def test_eta_zero_is_uniform_inside_each_parent(self):
        artifact, thought, action = materialize(
            self.rows, estimator(), 0.0, 0.0, 0.0, "uniform", 0.95, 1.0
        )
        positive = next(row for row in thought if row["episode_return"] == 1.0)
        positive_action = next(row for row in action if row["episode_return"] == 1.0)
        hierarchy = next(
            row["hierarchy"] for row in artifact["episodes"] if row["return"] == 1.0
        )
        # One turn; its three children are Thought step 0, Thought step 1,
        # and joint Action.  Step 0 then splits across two Thought tokens.
        parent = self.advantage / 3.0
        self.assertAlmostEqual(positive["adv_map"][0], parent / 2.0, places=9)
        self.assertAlmostEqual(positive["adv_map"][1], parent / 2.0, places=9)
        self.assertAlmostEqual(positive["adv_map"][2], parent, places=9)
        self.assertAlmostEqual(positive_action["adv_map"][0], parent / 2.0, places=9)
        self.assertAlmostEqual(positive_action["adv_map"][1], parent / 2.0, places=9)
        self.assertAlmostEqual(hierarchy["episode_mass"], self.advantage, places=9)

    def test_group_advantage_is_recomputed_fail_closed(self):
        broken = copy.deepcopy(self.rows)
        broken[0]["group_advantage"] = 0.25
        with self.assertRaisesRegex(ValueError, "group advantage drift"):
            _episodes(broken)

    def test_episode_requires_complete_contiguous_turns(self):
        broken = copy.deepcopy(self.rows[:1])
        broken[0]["episode_horizon"] = 2
        with self.assertRaisesRegex(ValueError, "turn sequence/horizon"):
            _episodes(broken)

    def test_multitrace_rows_keep_native_thought_and_action_widths(self):
        artifact, thought, action = materialize(
            self.rows, estimator(), 0.5, 0.5, 0.5, "uniform", 0.95, 1.0
        )
        del artifact
        # The legacy online-credit builder predates the registered joint
        # token/position learner and therefore has no denoising-round credit
        # map.  Supply an explicit two-boundary fixture here; production
        # method-credit builders create this field themselves.
        for record in thought:
            record["step_credit_by_round"] = {
                "0": float(record["adv_map"][0] + record["adv_map"][1]),
                "1": float(record["adv_map"][2]),
            }
        with tempfile.TemporaryDirectory() as directory:
            thought_path = os.path.join(directory, "thought.json")
            action_path = os.path.join(directory, "action.json")
            with open(thought_path, "w", encoding="utf-8") as handle:
                json.dump(thought, handle)
            with open(action_path, "w", encoding="utf-8") as handle:
                json.dump(action, handle)
            native = build_native_rows(
                thought_path, action_path, mask_id=99,
                sample_order_seed=42001, thought_block_size=4,
            )
        self.assertTrue(any(row["source"] == "thought" for row in native))
        self.assertTrue(any(row["source"] == "action" for row in native))
        for row in native:
            self.assertEqual(
                len(row["extended_input_ids"]),
                row["start_pos"] + 2 * row["response_width"],
            )
            self.assertEqual(len(row["adv_map"]), row["response_width"])
            self.assertEqual(
                len(row["rollout_logp"]),
                row["start_pos"] + row["response_width"],
            )
            self.assertEqual(
                len(row["prediction_mask"]),
                row["start_pos"] + row["response_width"],
            )
            self.assertEqual(
                row["response_width"], 10 if row["source"] == "thought" else 2
            )
            self.assertEqual(row["block_size"], 4 if row["source"] == "thought" else 1)
            if row["source"] == "thought":
                self.assertEqual(row["attention_layout"], NATIVE_THOUGHT_SCORING_LAYOUT)
                selected = row["prediction_mask"][row["start_pos"] :]
                self.assertTrue(any(selected))
                self.assertFalse(any(selected[3:]))
                first_copy = row["extended_input_ids"][
                    row["start_pos"] : row["start_pos"] + row["response_width"]
                ]
                self.assertEqual(first_copy[6:], [99, 99, 99, 99])

    def test_native_thought_fixed_width_is_authenticated_and_block_aligned(self):
        valid = trace("thought", [21, 22, 23], [0, 0, 1])
        self.assertTrue(
            validate_decision_trace(
                valid, require_logprobs=True, require_sampling=True,
                exact_replay=False,
            )
        )
        broken = copy.deepcopy(valid)
        broken["native_replay"]["replay_width"] = 9
        body = dict(broken["native_replay"])
        body.pop("replay_sha256", None)
        broken["native_replay"]["replay_sha256"] = sha256_json(body)
        with self.assertRaisesRegex(ValueError, "fixed replay width is not block aligned"):
            validate_decision_trace(
                broken, require_logprobs=True, require_sampling=True,
                exact_replay=False,
            )

    def test_constrained_action_support_is_recorded_and_fail_closed(self):
        constrained = trace("action", [31, 32], [0, 1])
        constrained["sampling"] = sampling("trie")
        constrained["constraint_allowed_token_ids"] = [[30, 31], [32, 33]]
        constrained["decision_contract_version"] = "neda-v4-decision-trace-v1"
        self.assertTrue(
            validate_action_replay_support(constrained)
        )
        missing = copy.deepcopy(constrained)
        del missing["constraint_allowed_token_ids"]
        with self.assertRaisesRegex(ValueError, "recorded trie allowed-token"):
            validate_action_replay_support(missing)

        artifact, thought, action = materialize(
            self.rows, estimator(), 0.5, 0.5, 0.5, "uniform", 0.95, 1.0
        )
        del artifact
        for record in action:
            action_trace = record["decision_traces"]["action"]
            action_trace["sampling"] = sampling("trie")
            action_trace["decision_contract_version"] = "neda-v4-decision-trace-v1"
            action_trace["constraint_allowed_token_ids"] = [
                [token, token + 100] for token in action_trace["response_ids"]
            ]
        with tempfile.TemporaryDirectory() as directory:
            thought_path = os.path.join(directory, "thought.json")
            action_path = os.path.join(directory, "action.json")
            with open(thought_path, "w", encoding="utf-8") as handle:
                json.dump(thought, handle)
            with open(action_path, "w", encoding="utf-8") as handle:
                json.dump(action, handle)
            native = build_native_rows(
                thought_path, action_path, mask_id=99,
                sample_order_seed=42001, thought_block_size=4,
            )
        action_native = [row for row in native if row["source"] == "action"]
        self.assertTrue(action_native)
        self.assertEqual(action_native[0]["constraint_allowed_token_ids"][0], [31, 131])

    def test_distributed_native_rows_are_not_split_inside_a_row(self):
        plan = distributed_step_plan(65, world_size=8, accumulation=2, epochs=1)
        self.assertEqual(plan["local_microbatches_per_epoch"], 9)
        self.assertEqual(plan["padded_global_microbatches_per_epoch"], 72)
        self.assertEqual(plan["optimizer_steps_per_epoch"], 5)
        self.assertEqual(plan["expected_optimizer_steps"], 5)
        with self.assertRaisesRegex(ValueError, "must all be positive"):
            distributed_step_plan(0, world_size=8, accumulation=2, epochs=1)

    def test_distributed_padding_is_explicit_unique_and_zero_credit(self):
        rows = [
            {
                "sample_id": "sample-{}".format(index),
                "source": "thought" if index % 2 == 0 else "action",
                "round_id": index,
                "extended_input_ids": [1, 2, 3 + index],
                "adv_map": [0.25, -0.25],
                "step_credit": 0.5 if index % 2 == 0 else None,
            }
            for index in range(5)
        ]
        padded, summary = pad_native_rows_for_distributed(
            rows, world_size=4, accumulation=2
        )
        self.assertEqual(summary["unpadded_rows"], 5)
        self.assertEqual(summary["padding_rows"], 3)
        self.assertEqual(summary["padded_rows"], 8)
        self.assertEqual(summary["padding_quantum"], 8)
        self.assertEqual(len(padded), 8)
        self.assertEqual(len({row["sample_id"] for row in padded}), 8)
        self.assertTrue(
            all(not row["is_distributed_padding"] for row in padded[:5])
        )
        for row in padded[5:]:
            self.assertTrue(row["is_distributed_padding"])
            self.assertEqual(row["adv_map"], [0.0, 0.0])
            if row["step_credit"] is not None:
                self.assertEqual(row["step_credit"], 0.0)
        # The input remains an immutable scientific row set.
        self.assertNotIn("is_distributed_padding", rows[0])
        self.assertEqual(rows[0]["adv_map"], [0.25, -0.25])


if __name__ == "__main__":
    unittest.main()
