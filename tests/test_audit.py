import unittest
from contextlib import contextmanager

from src import audit as audit_module
from src.audit import persist_audit_record
from src.db import Base
from src.db_models import AuditLog, Customer, RecoveryAttempt


def sqlite_session_factory():
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


def sample_result():
    return {
        "event_id": "evt-audit-1",
        "record_id": "rec-audit-1",
        "payment_id": "pay-audit-1",
        "customer_id": "cust-audit-1",
        "ml_suggested_action": "retry_24h",
        "final_action": "retry_24h",
        "policy_status": "allowed",
        "policy_reason": "ok",
        "execution_status": "executed",
        "audit_record": {"record_id": "rec-audit-1"},
    }


class AuditTests(unittest.TestCase):
    def test_persist_writes_attempt_log_and_bumps_counter(self):
        Session, engine, fake_get_session = sqlite_session_factory()
        try:
            session = Session()
            try:
                session.add(Customer(customer_id="cust-audit-1"))
                session.commit()
            finally:
                session.close()

            orig = audit_module.get_session
            audit_module.get_session = fake_get_session
            try:
                persist_audit_record(sample_result())
            finally:
                audit_module.get_session = orig

            session = Session()
            try:
                self.assertEqual(session.query(RecoveryAttempt).count(), 1)
                self.assertEqual(session.query(AuditLog).count(), 1)
                customer = session.get(Customer, "cust-audit-1")
                self.assertEqual(customer.previous_failed_payments, 1)
                self.assertEqual(customer.previous_recovered_payments, 0)
            finally:
                session.close()
        finally:
            Base.metadata.drop_all(bind=engine)
            engine.dispose()

    def test_failure_rolls_back_attempt_and_log_together(self):
        Session, engine, fake_get_session = sqlite_session_factory()
        try:
            session = Session()
            try:
                session.add(Customer(customer_id="cust-audit-1"))
                session.commit()
            finally:
                session.close()

            orig_session = audit_module.get_session
            orig_write = audit_module.repository.write_audit_log
            audit_module.get_session = fake_get_session
            audit_module.repository.write_audit_log = lambda s, r: (_ for _ in ()).throw(RuntimeError("boom"))
            try:
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    persist_audit_record(sample_result())
            finally:
                audit_module.get_session = orig_session
                audit_module.repository.write_audit_log = orig_write

            session = Session()
            try:
                self.assertEqual(session.query(RecoveryAttempt).count(), 0)
                self.assertEqual(session.query(AuditLog).count(), 0)
                customer = session.get(Customer, "cust-audit-1")
                self.assertEqual(customer.previous_failed_payments, 0)
            finally:
                session.close()
        finally:
            Base.metadata.drop_all(bind=engine)
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
