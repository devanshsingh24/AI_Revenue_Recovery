from types import SimpleNamespace
import unittest

from src import repository
from src.db_models import Customer, Payment, WebhookEvent


class FakeSession:
    def __init__(self):
        self.rows = {}
        self.added = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def get(self, model, identifier):
        return self.rows.get((model, identifier))

    def add(self, row):
        self.added.append(row)
        if isinstance(row, WebhookEvent):
            self.rows[(WebhookEvent, row.event_id)] = row

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class RepositoryTests(unittest.TestCase):
    def test_customer_feature_mapping(self):
        session = FakeSession()
        session.rows[(Customer, "customer-1")] = SimpleNamespace(
            segment="smb", tenure_months=4, past_payment_success_rate=0.7,
            historical_engagement_score=0.4, previous_failed_payments=2,
            previous_recovered_payments=1,
        )
        features = repository.get_customer_features(session, "customer-1")
        self.assertEqual(features["customer_segment"], "smb")
        self.assertEqual(features["previous_failed_payments"], 2)

    def test_payment_is_recorded_once(self):
        session = FakeSession()
        repository.record_payment(session, payment_id="payment-1", customer_id=None,
                                  order_id=None, amount=100.0, currency="INR", decline_reason="bank_decline")
        self.assertEqual(len(session.added), 1)
        self.assertIsInstance(session.added[0], Payment)

    def test_idempotency_store_commits_new_event_and_returns_cached_response(self):
        session = FakeSession()
        store = repository.PostgresIdempotencyStore(lambda: session)
        response = {"status": "processed"}

        store.set("event-1", response)

        self.assertEqual(session.commits, 1)
        self.assertTrue(session.closed)
        self.assertEqual(store.get("event-1"), response)


class SqliteRepositoryTests(unittest.TestCase):
    """Real-SQLAlchemy coverage without requiring PostgreSQL."""

    def setUp(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from src.db import Base

        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.addCleanup(Base.metadata.drop_all, bind=self.engine)
        self.addCleanup(self.engine.dispose)

    def test_upsert_and_feature_roundtrip(self):
        session = self.Session()
        try:
            self.assertIsNone(repository.get_customer_features(session, "cust-sql-1"))
            repository.upsert_customer_from_notes(session, "cust-sql-1", {"customer_segment": "smb"})
            session.commit()
            features = repository.get_customer_features(session, "cust-sql-1")
            self.assertEqual(features["customer_segment"], "smb")
            # Second upsert must not overwrite the existing row.
            repository.upsert_customer_from_notes(session, "cust-sql-1", {"customer_segment": "enterprise"})
            session.commit()
            self.assertEqual(
                repository.get_customer_features(session, "cust-sql-1")["customer_segment"], "smb"
            )
        finally:
            session.close()

    def test_recovery_attempt_bumps_failed_only_and_audit_persists(self):
        session = self.Session()
        try:
            repository.upsert_customer_from_notes(session, "cust-sql-2", {})
            repository.record_payment(
                session, payment_id="pay-sql-1", customer_id="cust-sql-2",
                order_id="order-sql-1", amount=500.0, currency="INR",
                decline_reason="bank_decline",
            )
            session.commit()
            repository.record_recovery_attempt(session, {
                "payment_id": "pay-sql-1", "customer_id": "cust-sql-2",
                "ml_suggested_action": "retry_24h", "final_action": "retry_24h",
                "policy_status": "allowed", "policy_reason": "ok",
                "execution_status": "executed",
            })
            repository.write_audit_log(session, {
                "event_id": "evt-sql-1", "record_id": "rec-sql-1",
                "payment_id": "pay-sql-1",
                "audit_record": {"record_id": "rec-sql-1"},
            })
            session.commit()
            customer = session.get(Customer, "cust-sql-2")
            self.assertEqual(customer.previous_failed_payments, 1)
            self.assertEqual(customer.previous_recovered_payments, 0)
        finally:
            session.close()

    def test_idempotency_store_roundtrip_on_real_session(self):
        store = repository.PostgresIdempotencyStore(self.Session)
        response = {"status": "processed", "payment_id": "pay-sql-1"}
        self.assertIsNone(store.get("evt-sql-2"))
        store.set("evt-sql-2", response)
        self.assertEqual(store.get("evt-sql-2"), response)


if __name__ == "__main__":
    unittest.main()
