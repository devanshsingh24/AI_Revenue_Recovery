import hashlib
import hmac
import json
import unittest
from contextlib import contextmanager

from src import audit as audit_module
from src import webhook_handler as wh_module
from src.agent.graph import build_graph
from src.db import Base
from src.db_models import AuditLog, Customer, Payment, RecoveryAttempt
from src.repository import PostgresIdempotencyStore
from src.webhook_handler import WebhookHandler
from train_policy import FEATURE_COLS, load_model


def make_sqlite_stack():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    @contextmanager
    def fake_get_session():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return Session, engine, fake_get_session


def signed_body(payload, secret="test-webhook-secret"):
    raw = json.dumps(payload, separators=(",", ":")).encode()
    sig = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return raw, sig


class IntegrationTests(unittest.TestCase):
    def test_webhook_to_graph_to_dryrun_to_sqlite(self):
        Session, engine, fake_get_session = make_sqlite_stack()
        try:
            from src.razorpay_client import RazorpayClient

            client = RazorpayClient(dry_run=True, webhook_secret="test-webhook-secret")
            model, feature_cols = load_model()
            assert feature_cols == FEATURE_COLS
            graph = build_graph(model, FEATURE_COLS, client)
            store = PostgresIdempotencyStore(Session)
            handler = WebhookHandler(graph, client, store)

            orig_wh_session = wh_module.get_session
            orig_audit_session = audit_module.get_session
            wh_module.get_session = fake_get_session
            audit_module.get_session = fake_get_session
            try:
                payload = {
                    "id": "evt-int-1",
                    "event": "payment.failed",
                    "payload": {"payment": {"entity": {
                        "id": "pay-int-1",
                        "order_id": "order-int-1",
                        "amount": 180000,
                        "currency": "INR",
                        "method": "card",
                        "error_reason": "insufficient_funds",
                        "notes": {"record_id": "REC-INT-1", "customer_id": "CUST-INT-1"},
                    }}},
                }
                raw, sig = signed_body(payload)
                out = handler.handle(raw, {"x-razorpay-signature": sig})
                self.assertEqual(out["status"], "processed")
                self.assertEqual(out["payment_id"], "pay-int-1")
                self.assertIn(out["final_action"], {
                    "retry_2h", "retry_24h", "payment_link", "escalation", "stop"})

                session = Session()
                try:
                    self.assertIsNotNone(session.get(Payment, "pay-int-1"))
                    self.assertIsNotNone(session.get(Customer, "CUST-INT-1"))
                    self.assertEqual(session.query(RecoveryAttempt).count(), 1)
                    self.assertEqual(session.query(AuditLog).count(), 1)
                    self.assertIsNotNone(store.get("evt-int-1"))
                finally:
                    session.close()

                replay = handler.handle(raw, {"x-razorpay-signature": sig})
                self.assertEqual(replay["status"], "duplicate_ignored")
                session = Session()
                try:
                    self.assertEqual(session.query(RecoveryAttempt).count(), 1)
                finally:
                    session.close()
            finally:
                wh_module.get_session = orig_wh_session
                audit_module.get_session = orig_audit_session
        finally:
            Base.metadata.drop_all(bind=engine)
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
