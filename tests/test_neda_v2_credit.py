import copy
import json
import math
import os
import tempfile
import unittest

from neda_joint_credit_v2 import materialize, neda_v2_weights
from neda_joint_ablation import materialize_variant
from neda_repro import sha256_json
from neda_v2_decision import build_legal_action_margin_evidence
from neda_v4_multitrace import _load_records
from tests.test_neda_v4_online_credit import row


def attach_v2_evidence(value, token_rows):
    result = copy.deepcopy(value)
    thought = result["decision_traces"]["thought"]
    action = result["decision_traces"]["action"]
    action["constraint_allowed_token_ids"] = [[31, 41], [32, 42]]
    action["sampling"]["constraint"] = "trie"
    action["decision_contract_version"] = "neda-v4-decision-trace-v1"
    action_body = dict(action)
    action_body.pop("trace_sha256", None)
    action["trace_sha256"] = sha256_json(action_body)
    v1 = {
        "contract_version": "neda-action-evidence-v1",
        "boundaries": [0, 1],
        "sequence_logprobs": [
            sum(values) / len(values) for values in token_rows
        ],
        "token_logprobs": token_rows,
        "segments": [
            {
                "segment_id": 0,
                "left_round_exclusive": -1,
                "right_round_inclusive": 0,
                "member_rounds": [0],
                "action_logprob_delta": 0.0,
            },
            {
                "segment_id": 1,
                "left_round_exclusive": 0,
                "right_round_inclusive": 1,
                "member_rounds": [1],
                "action_logprob_delta": 0.0,
            },
        ],
    }
    thought["action_evidence_v2"] = build_legal_action_margin_evidence(
        v1, action
    )
    return result


class NeDAV2CreditTest(unittest.TestCase):
    def test_margin_uses_exact_other_legal_mass(self):
        value = row("episode-a", 0, 1.0, 1.0)
        value = attach_v2_evidence(
            value,
            [
                [math.log(0.5), math.log(0.5)],
                [math.log(0.8), math.log(0.5)],
                [math.log(0.9), math.log(0.8)],
            ],
        )
        evidence = value["decision_traces"]["thought"]["action_evidence_v2"]
        self.assertAlmostEqual(
            evidence["executed_action_logprobs"][0], math.log(0.25)
        )
        self.assertAlmostEqual(
            evidence["other_legal_action_logprobs"][0], math.log(0.75)
        )
        self.assertGreater(
            evidence["legal_action_margin_delta"][0]
            if "legal_action_margin_delta" in evidence
            else evidence["segments"][0]["legal_action_margin_delta"],
            0.0,
        )

    def test_softmax_prefers_the_larger_margin_gain_and_conserves_mass(self):
        value = row("episode-a", 0, 1.0, 1.0)
        value = attach_v2_evidence(
            value,
            [
                [math.log(0.5), math.log(0.5)],
                [math.log(0.6), math.log(0.5)],
                [math.log(0.95), math.log(0.8)],
            ],
        )
        weights, diagnostic = neda_v2_weights(
            value["decision_traces"]["thought"],
            interpolation=0.5,
            evidence_temperature=0.25,
            boundaries_per_thought=2,
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights[1], weights[0])
        self.assertAlmostEqual(
            sum(diagnostic["evidence_weights"].values()), 1.0
        )

    def test_materializer_keeps_episode_and_turn_mass(self):
        advantage = 0.5 / 0.500001
        positive = attach_v2_evidence(
            row("episode-pos", 0, 1.0, advantage),
            [[-0.7, -0.7], [-0.5, -0.5], [-0.2, -0.2]],
        )
        negative = attach_v2_evidence(
            row("episode-neg", 1, 0.0, -advantage),
            [[-0.7, -0.7], [-0.5, -0.5], [-0.2, -0.2]],
        )
        artifact, thought, action = materialize(
            [positive, negative],
            interpolation=0.5,
            evidence_temperature=0.25,
            boundaries_per_thought=2,
        )
        self.assertEqual(artifact["method_id"], "neda_v2")
        self.assertEqual(artifact["n_nonzero_episodes"], 2)
        for record in thought:
            self.assertAlmostEqual(
                sum(record["step_credit_by_round"].values()),
                record["reward"],
            )
            self.assertEqual(record["registered_method"], "neda")
            self.assertEqual(record["method_variant"], "neda_v2")
            self.assertEqual(
                record["method_credit_contract"],
                "neda-v2-joint-credit-v1",
            )
        self.assertEqual(len(action), 2)
        with tempfile.TemporaryDirectory() as directory:
            thought_path = os.path.join(directory, "thought.json")
            action_path = os.path.join(directory, "action.json")
            with open(thought_path, "w", encoding="utf-8") as handle:
                json.dump(thought, handle)
            with open(action_path, "w", encoding="utf-8") as handle:
                json.dump(action, handle)
            loaded_thought = _load_records(thought_path, "thought")
            loaded_action = _load_records(action_path, "action")
        self.assertEqual(len(loaded_thought), len(thought))
        self.assertEqual(len(loaded_action), len(action))
        self.assertTrue(
            all(
                item["credit_contract"] == "neda-v2-joint-credit-v1"
                for item in loaded_thought + loaded_action
            )
        )

        for variant in (
            "uniform",
            "evidence_only",
            "no_position",
            "token_only",
            "no_horizon",
        ):
            out, out_thought, out_action, receipt = materialize_variant(
                variant,
                artifact,
                thought,
                action,
                activate_online=True,
                online_iteration=0,
            )
            self.assertEqual(out["ablation_id"], variant)
            self.assertEqual(receipt["status"], "PASS")
            self.assertTrue(receipt["scientific_result"])
            self.assertEqual(len(out_thought), len(thought))
            self.assertEqual(len(out_action), len(action))


if __name__ == "__main__":
    unittest.main()
