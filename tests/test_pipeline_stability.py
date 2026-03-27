import unittest
from unittest.mock import patch

from core.edit_engine import _build_lyaplock_defaults
from core.model_backend import ModelManager, apply_real_edit
from core.pipeline_runner import PipelineRunner


class PipelineStabilityTests(unittest.TestCase):
    def test_build_lyaplock_defaults_uses_layer_ids(self):
        defaults = _build_lyaplock_defaults([17, 23])
        self.assertEqual(set(defaults["V"].keys()), {17, 23})
        self.assertEqual(defaults["cnt"], 0)

    def test_pipeline_runner_model_ready_gate(self):
        runner = PipelineRunner(emit_fn=lambda _: None)
        ok, _ = runner._is_model_ready()
        self.assertFalse(ok)

        runner.active_model_name = "gpt2"
        runner.backend.model = object()
        runner.backend.tokenizer = object()
        ok, _ = runner._is_model_ready()
        self.assertTrue(ok)

    @patch("core.edit_engine.ROMEEditEngine.apply_edit")
    def test_apply_real_edit_normalizes_status_contract(self, mock_apply_edit):
        manager = ModelManager()
        manager.model = object()
        manager.tokenizer = object()

        mock_apply_edit.return_value = type(
            "FakeResult",
            (),
            {
                "success": True,
                "method": "ROME_only",
                "weights_changed": ["w1"],
                "error_message": "",
                "notes": "ok",
            },
        )()

        out = apply_real_edit(
            model_manager=manager,
            subject="Egypt",
            prompt_template="The president of {} is",
            target_new="Abdel Fattah el-Sisi",
            target_old="Hosni Mubarak",
            layer_hint=17,
            neuron_hint=100,
        )
        self.assertEqual(out["method"], "ROME_success_LyapLock_failed")
        self.assertTrue(out["success"])
        self.assertIn("post_checks", out)


if __name__ == "__main__":
    unittest.main()
