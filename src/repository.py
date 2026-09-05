from __future__ import annotations
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session
from src.db_models import Customer, Payment, RecoveryAttempt, AuditLog, WebhookEvent


def get_customer_features(session: Session, customer_id: str) -> Optional[Dict[str, Any]]:
    """The DB-first feature lookup your critique asked for. Returns None
    for a customer we've never recorded — caller decides the fallback,
    this function doesn't guess."""
    customer = session.get(Customer, customer_id)
    if customer is None:
        return None
    return {
        "customer_segment": customer.segment,
        "customer_tenure_months": customer.tenure_months,
        "past_payment_success_rate": customer.past_payment_success_rate,
        "historical_engagement_score": customer.historical_engagement_score,
        "previous_failed_payments": customer.previous_failed_payments,
        "previous_recovered_payments": customer.previous_recovered_payments,
    }


def upsert_customer_from_notes(session: Session, customer_id: str, notes: Dict[str, Any]) -> None:
    """Cold-start path: first time we see a customer_id, seed a row from
    whatever the webhook notes carried (or defaults). Every subsequent
    webhook for this customer_id reads from the DB row instead, so notes
    only matter once."""
    if session.get(Customer, customer_id) is not None:
        return
    session.add(Customer(
        customer_id=customer_id,
        segment=notes.get("customer_segment", "individual"),
        tenure_months=int(notes.get("customer_tenure_months", 0)),
        past_payment_success_rate=float(notes.get("past_payment_success_rate", 0.5)),
        historical_engagement_score=float(notes.get("historical_engagement_score", 0.5)),
        previous_failed_payments=int(notes.get("previous_failed_payments", 0)),
        previous_recovered_payments=int(notes.get("previous_recovered_payments", 0)),
    ))


def record_payment(session: Session, *, payment_id: str, customer_id: Optional[str],
                    order_id: Optional[str], amount: float, currency: str,
                    decline_reason: Optional[str]) -> None:
    if payment_id and session.get(Payment, payment_id) is None:
        session.add(Payment(payment_id=payment_id, customer_id=customer_id, order_id=order_id,
                             amount=amount, currency=currency, decline_reason=decline_reason))




def record_recovery_attempt(session: Session, result: Dict[str, Any]) -> None:
    session.add(RecoveryAttempt(
        payment_id=result.get("payment_id"),
        ml_suggested_action=result.get("ml_suggested_action"),
        final_action=result.get("final_action"),
        policy_status=result.get("policy_status"),
        policy_reason=result.get("policy_reason"),
        execution_status=result.get("execution_status"),
    ))

    # Only previous_failed_payments is updated here: a payment.failed event
    # unambiguously means one more failure happened. previous_recovered_payments
    # is intentionally NOT touched — execution_status == "executed" means an
    # order/payment-link was created, not that the customer paid. There is
    # no payment.captured/order.paid handler in this codebase yet, so there
    # is no true signal for "recovered." Incrementing it here would write a
    # false success into the same customer table the ML model reads features
    # from at inference time. Leave it at whatever value seeded it (from
    # webhook notes on first sighting) until a real capture-confirmation
    # webhook exists to update it correctly.
    customer_id = result.get("customer_id")
    if customer_id:
        customer = session.get(Customer, customer_id)
        if customer is not None:
            customer.previous_failed_payments += 1

def write_audit_log(session: Session, result: Dict[str, Any]) -> None:
    audit_record = result.get("audit_record") or {}
    session.add(AuditLog(
        event_id=result.get("event_id"),
        record_id=result.get("record_id") or audit_record.get("record_id"),
        payment_id=result.get("payment_id"),
        record=audit_record,
    ))


class PostgresIdempotencyStore:
    """Concrete implementation of the IdempotencyStore protocol already
    defined in webhook_handler.py. Replaces the in-memory version —
    survives restarts and works across multiple app workers, since both
    read/write the same `webhook_events` table."""
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def get(self, event_id: str) -> Optional[Dict[str, Any]]:
        session = self._session_factory()
        try:
            row = session.get(WebhookEvent, event_id)
            return row.response if row else None
        finally:
            session.close()

    def set(self, event_id: str, result: Dict[str, Any]) -> None:
        session = self._session_factory()
        try:
            if session.get(WebhookEvent, event_id) is None:
                session.add(WebhookEvent(event_id=event_id, response=result))
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
