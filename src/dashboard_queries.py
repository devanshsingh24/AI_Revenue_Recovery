"""Read-only dashboard queries over the EXISTING schema.

Tables used (all pre-existing, see src/db_models.py):
customers, payments, recovery_attempts, audit_logs, webhook_events.

Data-honesty rule: this system has NO capture-confirmation signal.
``execution_status == "executed"`` means an order/payment-link was CREATED,
not that money arrived. Nothing here computes "recovered amount" or a
"recovery rate" — amount sums are labeled "amount attempted" by callers.

Every function takes a live SQLAlchemy session and optional ``start``/``end``
datetimes filtering that query's primary ``created_at`` column. All functions
are SELECT-only; this module never writes.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select

from src.db_models import AuditLog, Payment, RecoveryAttempt, WebhookEvent


def _in_range(column, start: Optional[datetime], end: Optional[datetime]):
    filters = []
    if start is not None:
        filters.append(column >= start)
    if end is not None:
        filters.append(column <= end)
    return filters


def get_total_failed_payments(session, start: Optional[datetime] = None,
                              end: Optional[datetime] = None) -> int:
    stmt = select(func.count()).select_from(Payment).where(
        *_in_range(Payment.created_at, start, end))
    return int(session.execute(stmt).scalar() or 0)


def get_final_action_counts(session, start: Optional[datetime] = None,
                            end: Optional[datetime] = None) -> Dict[str, int]:
    stmt = (select(RecoveryAttempt.final_action, func.count())
            .where(*_in_range(RecoveryAttempt.created_at, start, end))
            .group_by(RecoveryAttempt.final_action))
    return {(row[0] or "unknown"): int(row[1]) for row in session.execute(stmt).all()}


def get_policy_status_counts(session, start: Optional[datetime] = None,
                             end: Optional[datetime] = None) -> Dict[str, int]:
    stmt = (select(RecoveryAttempt.policy_status, func.count())
            .where(*_in_range(RecoveryAttempt.created_at, start, end))
            .group_by(RecoveryAttempt.policy_status))
    return {(row[0] or "unknown"): int(row[1]) for row in session.execute(stmt).all()}


def get_execution_status_counts(session, start: Optional[datetime] = None,
                                end: Optional[datetime] = None) -> Dict[str, int]:
    stmt = (select(RecoveryAttempt.execution_status, func.count())
            .where(*_in_range(RecoveryAttempt.created_at, start, end))
            .group_by(RecoveryAttempt.execution_status))
    return {(row[0] or "unknown"): int(row[1]) for row in session.execute(stmt).all()}


def get_decline_reason_breakdown(session, start: Optional[datetime] = None,
                                 end: Optional[datetime] = None) -> Dict[str, int]:
    stmt = (select(Payment.decline_reason, func.count())
            .where(*_in_range(Payment.created_at, start, end))
            .group_by(Payment.decline_reason))
    return {(row[0] or "unknown"): int(row[1]) for row in session.execute(stmt).all()}


def get_amount_attempted_by_final_action(
        session, start: Optional[datetime] = None,
        end: Optional[datetime] = None) -> Dict[str, float]:
    """Sum of payments.amount grouped by final_action.

    "Amount attempted" — the value at risk when the action was taken.
    NEVER label this "amount recovered" (no capture signal exists).
    """
    stmt = (select(RecoveryAttempt.final_action, func.sum(Payment.amount))
            .join(Payment, Payment.payment_id == RecoveryAttempt.payment_id)
            .where(*_in_range(Payment.created_at, start, end))
            .group_by(RecoveryAttempt.final_action))
    return {(row[0] or "unknown"): float(row[1] or 0.0) for row in session.execute(stmt).all()}


def get_recovery_detail_rows(session, start: Optional[datetime] = None,
                             end: Optional[datetime] = None,
                             limit: int = 500) -> List[Dict[str, Any]]:
    """Flat rows for the Recovery Detail table (payments LEFT JOIN attempts)."""
    stmt = (select(Payment, RecoveryAttempt)
            .outerjoin(RecoveryAttempt,
                       RecoveryAttempt.payment_id == Payment.payment_id)
            .where(*_in_range(Payment.created_at, start, end))
            .order_by(Payment.created_at.desc())
            .limit(limit))
    rows = []
    for payment, attempt in session.execute(stmt).all():
        rows.append({
            "payment_id": payment.payment_id,
            "customer_id": payment.customer_id,
            "decline_reason": payment.decline_reason,
            "amount": payment.amount,
            "currency": payment.currency,
            "ml_suggested_action": getattr(attempt, "ml_suggested_action", None),
            "final_action": getattr(attempt, "final_action", None),
            "policy_status": getattr(attempt, "policy_status", None),
            "policy_reason": getattr(attempt, "policy_reason", None),
            "execution_status": getattr(attempt, "execution_status", None),
            "created_at": payment.created_at,
        })
    return rows


def get_payment_detail(session, payment_id: str) -> Dict[str, Any]:
    """Single-record drill-down: payment + latest attempt + latest audit record."""
    payment = session.get(Payment, payment_id)
    if payment is None:
        return {}
    attempt_stmt = (select(RecoveryAttempt)
                    .where(RecoveryAttempt.payment_id == payment_id)
                    .order_by(RecoveryAttempt.id.desc()).limit(1))
    attempt = session.execute(attempt_stmt).scalars().first()
    audit_stmt = (select(AuditLog)
                  .where(AuditLog.payment_id == payment_id)
                  .order_by(AuditLog.id.desc()).limit(1))
    audit = session.execute(audit_stmt).scalars().first()
    record = dict(getattr(audit, "record", None) or {})
    return {
        "payment_id": payment.payment_id,
        "customer_id": payment.customer_id,
        "order_id": payment.order_id,
        "amount": payment.amount,
        "currency": payment.currency,
        "decline_reason": payment.decline_reason,
        "created_at": payment.created_at,
        "ml_suggested_action": getattr(attempt, "ml_suggested_action", None),
        "final_action": getattr(attempt, "final_action", None),
        "policy_status": getattr(attempt, "policy_status", None),
        "policy_reason": getattr(attempt, "policy_reason", None),
        "execution_status": getattr(attempt, "execution_status", None),
        "audit_record": record,
        "diagnosis": record.get("diagnosis") or {},
        "action_probabilities": record.get("action_probabilities") or {},
        "expected_net_recovery": record.get("expected_net_recovery") or {},
        "execution_result": record.get("execution_result") or {},
        "escalation_result": record.get("escalation_result") or {},
    }


def get_recent_webhook_events(session, limit: int = 100) -> List[Dict[str, Any]]:
    """Most recent webhook_events; status/payment extracted in Python.

    JSON is parsed in Python (not with DB-specific JSON operators) so this
    works on both PostgreSQL and SQLite.
    """
    stmt = (select(WebhookEvent)
            .order_by(WebhookEvent.created_at.desc()).limit(limit))
    out = []
    for event in session.execute(stmt).scalars().all():
        response = dict(getattr(event, "response", None) or {})
        out.append({
            "event_id": event.event_id,
            "received_at": event.created_at,
            "status": response.get("status", "unknown"),
            "payment_id": response.get("payment_id"),
        })
    return out


def get_model_confidence_stats(session, start: Optional[datetime] = None,
                               end: Optional[datetime] = None) -> Dict[str, Any]:
    """Average ml_confidence + mean action_probabilities from audit JSON.

    Parsed in Python for backend portability; malformed records are skipped.
    """
    stmt = select(AuditLog).where(*_in_range(AuditLog.created_at, start, end))
    confidences: List[float] = []
    prob_sums: Dict[str, float] = {}
    prob_counts: Dict[str, int] = {}
    examined = 0
    for audit in session.execute(stmt).scalars().all():
        record = getattr(audit, "record", None) or {}
        if not isinstance(record, dict):
            continue
        examined += 1
        try:
            conf = record.get("ml_confidence")
            if conf is not None:
                confidences.append(float(conf))
            probs = record.get("action_probabilities") or {}
            if isinstance(probs, dict):
                for action, value in probs.items():
                    prob_sums[action] = prob_sums.get(action, 0.0) + float(value)
                    prob_counts[action] = prob_counts.get(action, 0) + 1
        except (TypeError, ValueError):
            continue
    return {
        "records_examined": examined,
        "avg_ml_confidence": (sum(confidences) / len(confidences)) if confidences else None,
        "mean_action_probabilities": {
            action: prob_sums[action] / prob_counts[action]
            for action in prob_sums if prob_counts[action]
        },
    }


def get_audit_trail(session, payment_id: Optional[str] = None,
                    customer_id: Optional[str] = None,
                    limit: int = 50) -> List[Dict[str, Any]]:
    """Chronological merchant-readable story for one payment or customer."""
    trail: List[Dict[str, Any]] = []
    payment_ids: List[str] = []
    if payment_id:
        payment_ids = [payment_id]
    elif customer_id:
        stmt = (select(Payment.payment_id)
                .where(Payment.customer_id == customer_id)
                .order_by(Payment.created_at.desc()).limit(limit))
        payment_ids = [row[0] for row in session.execute(stmt).all()]
    for pid in payment_ids:
        detail = get_payment_detail(session, pid)
        if not detail:
            continue
        record = detail.get("audit_record") or {}
        trail.append({
            "payment_id": pid,
            "created_at": detail.get("created_at"),
            "steps": [
                {"stage": "Webhook received",
                 "summary": f"payment.failed for {pid} "
                            f"({detail.get('decline_reason')}, "
                            f"{detail.get('amount')} {detail.get('currency')})"},
                {"stage": "Diagnosis",
                 "summary": str((detail.get("diagnosis") or {}).get(
                     "classification", "classified failure"))},
                {"stage": "ML scores",
                 "summary": f"suggested {detail.get('ml_suggested_action')} "
                            f"(confidence {record.get('ml_confidence')})"},
                {"stage": "Policy decision",
                 "summary": f"{detail.get('policy_status')}: "
                            f"{detail.get('policy_reason')} → {detail.get('final_action')}"},
                {"stage": "Execution / escalation",
                 "summary": str(detail.get("execution_status"))},
                {"stage": "Audit record",
                 "summary": f"record {record.get('record_id') or pid} stored"},
            ],
            "detail": detail,
        })
    return trail
