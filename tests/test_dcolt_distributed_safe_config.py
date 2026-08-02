import json
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "accelerate_configs" / "dcolt_zero3_comm_safe.json"
CONFIG_CPU = ROOT / "accelerate_configs" / "dcolt_zero3_cpu_comm_safe.json"
CONFIG_1 = ROOT / "accelerate_configs" / "1_gpu_zero3_cpu_dcolt_safe.yaml"
CONFIG_4 = (
    ROOT / "accelerate_configs" / "1_node_4_gpus_zero3_dcolt_safe.yaml"
)
CONFIG_8 = (
    ROOT / "accelerate_configs" / "1_node_8_gpus_zero3_dcolt_safe.yaml"
)
LAUNCHER = ROOT / "train" / "rl_sdar_multitrace_dcolt_safe.py"
LEARNER = ROOT / "train" / "rl_sdar_multitrace.py"


class DCoLTDistributedSafeConfigTests(unittest.TestCase):
    def setUp(self):
        self.value = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.zero = self.value["zero_optimization"]

    def test_zero3_partitions_the_full_output_projection_parameter(self):
        self.assertEqual(self.zero["stage"], 3)
        self.assertTrue(self.zero["reduce_scatter"])
        self.assertFalse(self.zero["overlap_comm"])
        self.assertEqual(self.zero["reduce_bucket_size"], 25_000_000)
        self.assertEqual(self.zero["allgather_bucket_size"], 25_000_000)
        self.assertTrue(self.zero["stage3_gather_16bit_weights_on_model_save"])
        cpu = json.loads(CONFIG_CPU.read_text(encoding="utf-8"))["zero_optimization"]
        self.assertEqual(cpu["offload_optimizer"]["device"], "cpu")
        self.assertEqual(cpu["offload_param"]["device"], "cpu")

    def test_configs_keep_world_size_and_share_one_deepspeed_contract(self):
        for path, world_size, expected_path in (
            (CONFIG_1, 1, "accelerate_configs/dcolt_zero3_cpu_comm_safe.json"),
            (CONFIG_4, 4, "accelerate_configs/dcolt_zero3_comm_safe.json"),
            (CONFIG_8, 8, "accelerate_configs/dcolt_zero3_comm_safe.json"),
        ):
            text = path.read_text(encoding="utf-8")
            self.assertIn(f"num_processes: {world_size}", text)
            self.assertIn(f"deepspeed_config_file: {expected_path}", text)
            self.assertNotIn("zero_stage:", text)
            self.assertNotIn("offload_optimizer_device:", text)
            self.assertNotIn("mixed_precision:", text)
        self.assertEqual(self.value["bf16"]["enabled"], "auto")

    def test_observed_full_vocab_collective_is_partitioned(self):
        full_vocab_gradient = 151_936 * 4_096
        self.assertEqual(full_vocab_gradient, 622_329_856)
        self.assertLessEqual(
            (full_vocab_gradient + 4 - 1) // 4,
            155_582_464,
        )
        self.assertLessEqual(
            (full_vocab_gradient + 8 - 1) // 8,
            77_791_232,
        )

    def test_timeout_is_installed_at_first_accelerator_construction(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("training.registered_method=dcolt", text)
        learner = LEARNER.read_text(encoding="utf-8")
        self.assertIn('NEDA_DCOLT_PROCESS_GROUP_TIMEOUT_SECONDS", "3600"', learner)
        self.assertIn("InitProcessGroupKwargs", learner)
        self.assertIn('registered_method == "dcolt"', learner)

    def test_runner_passes_precision_on_cli_for_external_ds_config(self):
        runner = (
            ROOT.parent.parent
            / "qsub_bash_files"
            / "run_neda_joint_alfworld_sft.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "ACCELERATE_PRECISION_ARGS=(--mixed_precision bf16)", runner
        )
        self.assertIn(
            'accelerate launch "${ACCELERATE_PRECISION_ARGS[@]}"', runner
        )

    def test_dcolt_scores_in_behavior_eval_mode(self):
        text = LEARNER.read_text(encoding="utf-8")
        marker = "Rollout behavior scores are produced in eval mode"
        start = text.index(marker)
        no_grad = text.index("with torch.no_grad():", start)
        self.assertIn('registered_method == "dcolt"', text[start:no_grad])
        self.assertIn("model.eval()", text[start:no_grad])
        self.assertIn('"old_policy_scoring_mode": old_policy_scoring_mode', text)

    def test_every_native_row_traverses_the_upm_collective_graph(self):
        """Thought/Action rank mixing must not change ZeRO-3 collectives."""

        text = LEARNER.read_text(encoding="utf-8")
        self.assertIn('is_dcolt = registered_method == "dcolt"', text)
        self.assertIn("elif is_dcolt:", text)
        self.assertIn('"position_timestep": torch.zeros(', text)
        self.assertIn('"position_mask_index": dummy_mask', text)
        self.assertIn("outputs.position_logits.float().sum() * 0.0", text)
        self.assertIn(
            '"all-native-rows-upm-zero-anchor-v1"', text
        )


if __name__ == "__main__":
    unittest.main()
