import json
import unittest
from contextlib import contextmanager

from src import webhook_handler as wh_module
from src.webhook_handler import WebhookHandler


class FakeClient:
    def __init__(self, valid=True):
        self.valid = valid

    def verify_webhook_signature(self, raw_body, signature):
        return self.valid


class FakeStore:
    def __init__(self, cached=None):
        self.data = dict(cached or {})
        self.set_calls = []

    def get(self, event_id):
        return self.data.get(event_id)

    def set(self, event_id, result):
        self.set_calls.append((event_id, result))
        self.data[event_id] = result


def payment_failed_body(event_id="evt-1"):
    payload = {
        "id": event_id,
        "event": "payment.failed",
        "payload": {"payment": {"entity": {
            "id": "pay-1",
            "order_id": "order-1",
            "amount": 180000,
            "currency": "INR",
            "method": "card",
            "error_reason": "insufficient_funds",
            "notes": {"record_id": "REC-1"},
        }}},
    }
    return json.dumps(payload, separators=(",", ":")).encode()


class WebhookHandlerTests(unittest.TestCase):
    def make_handler(self, valid_sig=True, cached=None):
        store = FakeStore(cached)
        handler = WebhookHandler(
            compiled_graph=object(),
            razorpay_client=FakeClient(valid_sig),
            idempotency_store=store,
        )
        return handler, store

    def patch_deps(self, run_result=None):
        """Replace DB + graph boundaries with fakes. Returns recorders."""
        calls = {"run": [], "persist": [], "record_payment": []}
        result = run_result or {
            "payment_id": "pay-1",
            "ml_suggested_action": "retry_24h",
            "policy_status": "allowed",
            "policy_reason": "ok",
            "final_action": "retry_24h",
            "execution_status": "executed",
        }

        orig_run = wh_module.run_recovery
        orig_session = wh_module.get_session
        orig_persist = wh_module.persist_audit_record
        orig_record = wh_module.repository.record_payment
        orig_upsert = wh_module.repository.upsert_customer_from_notes
        orig_features = wh_module.repository.get_customer_features

        def fake_run(graph, state):
            calls["run"].append(state)
            return dict(result)

        @contextmanager
        def fake_session():
            yield object()

        def fake_record(session, **kwargs):
            calls["record_payment"].append(kwargs)

        wh_module.run_recovery = fake_run
        wh_module.get_session = fake_session
        wh_module.persist_audit_record = lambda r: calls["persist"].append(r)
        wh_module.repository.record_payment = fake_record
        wh_module.repository.upsert_customer_from_notes = lambda s, c, n: None
        wh_module.repository.get_customer_features = lambda s, c: {}

        self.addCleanup(setattr, wh_module, "run_recovery", orig_run)
        self.addCleanup(setattr, wh_module, "get_session", orig_session)
        self.addCleanup(setattr, wh_module, "persist_audit_record", orig_persist)
        self.addCleanup(setattr, wh_module.repository, "record_payment", orig_record)
        self.addCleanup(setattr, wh_module.repository, "upsert_customer_from_notes", orig_upsert)
        self.addCleanup(setattr, wh_module.repository, "get_customer_features", orig_features)
        return calls

    def test_missing_signature_is_rejected(self):
        handler, _ = self.make_handler()
        with self.assertRaisesRegex(ValueError, "Missing"):
            handler.handle(b"{}", {})

    def test_invalid_signature_is_rejected(self):
        handler, _ = self.make_handler(valid_sig=False)
        with self.assertRaisesRegex(ValueError, "Invalid"):
            handler.handle(b"{}", {"x-razorpay-signature": "bad"})

    def test_duplicate_event_skips_graph(self):
        cached = {"status": "processed", "payment_id": "pay-1"}
        handler, _ = self.make_handler(cached={"evt-1": cached})
        calls = self.patch_deps()
        out = handler.handle(payment_failed_body("evt-1"), {"x-razorpay-signature": "sig"})
        self.assertEqual(out["status"], "duplicate_ignored")
        self.assertEqual(out["payment_id"], "pay-1")
        self.assertEqual(calls["run"], [])

    def test_unsupported_event_is_ignored(self):
        handler, store = self.make_handler()
        calls = self.patch_deps()
        body = json.dumps({"id": "evt-9", "event": "payment.captured"}).encode()
        out = handler.handle(body, {"x-razorpay-signature": "sig"})
        self.assertEqual(out["status"], "ignored")
        self.assertEqual(calls["run"], [])
        self.assertEqual(calls["persist"], [])

    def test_payment_failed_is_processed_and_cached(self):
        handler, store = self.make_handler()
        calls = self.patch_deps()
        out = handler.handle(payment_failed_body("evt-1"), {"x-razorpay-signature": "sig"})
        self.assertEqual(out["status"], "processed")
        self.assertEqual(out["payment_id"], "pay-1")
        self.assertEqual(len(calls["run"]), 1)
        self.assertEqual(len(calls["persist"]), 1)
        self.assertEqual(len(calls["record_payment"]), 1)
        self.assertIn("evt-1", store.data)

    def test_graph_errors_yield_processed_with_errors(self):
        handler, _ = self.make_handler()
        calls = self.patch_deps(run_result={
            "payment_id": "pay-1", "errors": ["Boom"],
            "ml_suggested_action": "stop", "final_action": "stop",
            "policy_status": "blocked", "policy_reason": "x",
            "execution_status": "not_executed",
        })
        out = handler.handle(payment_failed_body("evt-err"), {"x-razorpay-signature": "sig"})
        self.assertEqual(out["status"], "processed_with_errors")
        self.assertEqual(out["errors"], ["Boom"])


if __name__ == "__main__":
    unittest.main()
