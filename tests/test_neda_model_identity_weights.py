import json
import os
import tempfile
import unittest
from pathlib import Path

from neda_repro import build_model_identity


class WeightAuthenticatedModelIdentityTest(unittest.TestCase):
    def _model(self, root: Path, payload: bytes) -> Path:
        root.mkdir()
        (root / "config.json").write_text("{}\n", encoding="utf-8")
        shard = root / "model-00001-of-00001.safetensors"
        shard.write_bytes(payload)
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": shard.name}}), encoding="utf-8"
        )
        import hashlib

        digest = hashlib.sha256(payload).hexdigest()
        (root / "WEIGHTS.SHA256SUMS").write_text(
            f"{digest}  {shard.name}\n", encoding="utf-8"
        )
        return root

    def test_equal_size_different_weights_have_different_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            left = build_model_identity(self._model(base / "left", b"left"))
            right = build_model_identity(self._model(base / "right", b"rght"))
            self.assertNotEqual(left["identity_sha256"], right["identity_sha256"])
            self.assertIn("sha256", left["weight_shards"][0])

    def test_required_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._model(Path(temporary) / "model", b"weights")
            (root / "WEIGHTS.SHA256SUMS").unlink()
            previous = os.environ.get("NEDA_REQUIRE_WEIGHT_HASH_IDENTITY")
            os.environ["NEDA_REQUIRE_WEIGHT_HASH_IDENTITY"] = "1"
            try:
                with self.assertRaises(FileNotFoundError):
                    build_model_identity(root)
            finally:
                if previous is None:
                    os.environ.pop("NEDA_REQUIRE_WEIGHT_HASH_IDENTITY", None)
                else:
                    os.environ["NEDA_REQUIRE_WEIGHT_HASH_IDENTITY"] = previous


if __name__ == "__main__":
    unittest.main()
