import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT.parent.parent / "qsub_bash_files" / "run_neda_joint_webshop.sh"
)


class WebShopDCoLTSafeRunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = RUNNER.read_text(encoding="utf-8")

    def test_dcolt_selects_safe_launcher_for_every_supported_world_size(self):
        for config in (
            "1_gpu_zero3_cpu_dcolt_safe.yaml",
            "1_node_4_gpus_zero3_dcolt_safe.yaml",
            "1_node_8_gpus_zero3_dcolt_safe.yaml",
        ):
            self.assertIn(config, self.runner)
        self.assertIn(
            'ACCELERATE_PRECISION_ARGS=(--mixed_precision bf16)',
            self.runner,
        )

    def test_launch_uses_selected_entry_and_precision_arguments(self):
        self.assertIn(
            'accelerate launch "${ACCELERATE_PRECISION_ARGS[@]}"',
            self.runner,
        )
        self.assertIn(
            '"$LEARNER_ENTRY" "${LEARNER_ARGS[@]}"',
            self.runner,
        )
        self.assertNotIn(
            '    train/rl_sdar_multitrace.py "${LEARNER_ARGS[@]}"',
            self.runner,
        )

    def test_non_dcolt_default_remains_the_original_learner(self):
        self.assertIn(
            'LEARNER_ENTRY="train/rl_sdar_multitrace.py"',
            self.runner,
        )
        self.assertIn('if [[ "$METHOD" == dcolt ]]; then', self.runner)

    def test_runner_has_quota_weight_seal_and_retention_gates(self):
        self.assertIn("NEDA_MIN_PROJECT_HEADROOM_GIB", self.runner)
        self.assertIn("WEIGHTS.SHA256SUMS", self.runner)
        self.assertIn("prune_checkpoint_before", self.runner)
        self.assertIn("verify_frozen_code", self.runner)


if __name__ == "__main__":
    unittest.main()
