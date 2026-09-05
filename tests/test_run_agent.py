import json
import unittest

import run_agent
import run_pipeline


class RunAgentTests(unittest.TestCase):
    def test_normalize_input_uses_notes_with_defaults(self):
        payload = {"id": "evt-1", "event": "payment.failed",
                   "payload": {"payment": {"entity": {
                       "id": "pay-1", "order_id": "order-1",
                       "amount": 180000, "currency": "INR",
                       "method": "card", "error_reason": "insufficient_funds",
                       "notes": {"record_id": "REC-1", "customer_id": "CUST1"},
                   }}}}
        state = run_agent.normalize_input(payload)
        self.assertEqual(state["payment_id"], "pay-1")
        self.assertEqual(state["amount"], 1800.0)
        self.assertEqual(state["customer_segment"], "individual")
        self.assertEqual(state["past_payment_success_rate"], 0.5)

    def test_main_persists_best_effort_and_prints(self):
        import run_agent as ra

        calls = {}
        orig_client = ra.RazorpayClient
        orig_build = ra.build_graph_from_trained_model
        orig_run = ra.run_recovery
        orig_persist = ra.persist_audit_record
        try:
            ra.RazorpayClient = lambda: object()
            ra.build_graph_from_trained_model = lambda client: object()
            ra.run_recovery = lambda graph, state: {
                "ml_suggested_action": "retry_24h", "policy_status": "allowed",
                "policy_reason": "ok", "final_action": "retry_24h",
                "execution_status": "executed", "errors": None,
                "audit_record": {"record_id": "REC-1"},
            }
            # Simulate DB down: persist must not crash the CLI.
            ra.persist_audit_record = lambda result: (_ for _ in ()).throw(RuntimeError("db down"))

            import io
            import tempfile
            from contextlib import redirect_stdout
            from pathlib import Path

            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump({"id": "evt-1", "event": "payment.failed",
                           "payload": {"payment": {"entity": {"id": "pay-1", "amount": 10000}}}}, f)
                payload_path = f.name
            buf = io.StringIO()
            with redirect_stdout(buf):
                ra.main(["--payload", payload_path])
            out = json.loads(buf.getvalue())
            self.assertEqual(out["final_action"], "retry_24h")
            self.assertIn("db down", out["persist_error"])
            Path(payload_path).unlink()
        finally:
            ra.RazorpayClient = orig_client
            ra.build_graph_from_trained_model = orig_build
            ra.run_recovery = orig_run
            ra.persist_audit_record = orig_persist

    def test_legacy_ml_policy_shim_scores_without_name_error(self):
        import pandas as pd
        from src.ml_policy import ml_policy
        from train_policy import ACTIONS, DATA_PATH

        row = pd.read_csv(DATA_PATH, nrows=1).iloc[0]
        out = ml_policy(row)
        self.assertIn(out["suggested_action"], ACTIONS)
        self.assertIn("all_action_scores", out)

    def test_execution_placeholder_is_importable(self):
        import src.execution  # noqa: F401 - must not raise

    def test_run_pipeline_batch_writes_audit_csv(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit.csv"
            result = run_pipeline.main(["--limit", "5", "--out", str(out)])
            self.assertEqual(result, out)
            self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
