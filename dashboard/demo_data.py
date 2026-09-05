"""Curated demo dataset for client walkthroughs (dashboard/demo_data.py).

Separate from production data: seeds a throwaway SQLite file in the OS temp
directory, so a demo never risks showing real customer information.

Honesty rule applies here too: seeded `execution_status == "executed"` rows
represent created orders/links, NOT confirmed receipts. Nothing here invents
a "recovered" outcome.

Dedup note: duplicate webhook deliveries intentionally leave NO second row
(the handler returns `duplicate_ignored` without writing). To show dedup in
a demo, replay one event live against the API and point at the response —
do not fake a duplicate row in this seed.
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

DEMO_DB_PATH = Path(tempfile.gettempdir()) / "rr_demo.db"

ACTIONS = ["retry_2h", "retry_24h", "payment_link", "escalation", "stop"]

# (payment_id, customer_id, decline_reason, amount, ml, final, policy, reason, execution, event_id, event_status, confidence)
DEMO_ROWS = [
    ("pay_demo_101", "demo_cust_01", "insufficient_funds", 4999.0,
     "retry_24h", "retry_24h", "allowed", "ML recommendation passed all policy checks.",
     "executed", "evt_demo_101", "processed", 0.73),
    ("pay_demo_102", "demo_cust_02", "technical_error", 1299.0,
     "retry_2h", "retry_2h", "allowed", "ML recommendation passed all policy checks.",
     "executed", "evt_demo_102", "processed", 0.81),
    ("pay_demo_103", "demo_cust_03", "card_expired", 8999.0,
     "retry_2h", "payment_link", "overridden", "Expired card requires customer instrument update.",
     "executed", "evt_demo_103", "processed", 0.66),
    ("pay_demo_104", "demo_cust_04", "disputed", 54999.0,
     "payment_link", "escalation", "overridden", "Disputed payment requires human review.",
     "escalated", "evt_demo_104", "processed", 0.58),
    ("pay_demo_105", "demo_cust_05", "fraud_flag", 7499.0,
     "retry_24h", "stop", "blocked", "Fraud/security signal: automatic recovery is prohibited.",
     "not_executed", "evt_demo_105", "processed", 0.62),
    ("pay_demo_106", "demo_cust_01", "bank_decline", 18999.0,
     "retry_24h", "retry_24h", "allowed", "ML recommendation passed all policy checks.",
     "executed", "evt_demo_106", "processed", 0.77),
    ("pay_demo_107", "demo_cust_06", "mandate_revoked", 3499.0,
     "retry_24h", "payment_link", "overridden", "Mandate revoked; bare retry is blocked.",
     "executed", "evt_demo_107", "processed", 0.69),
    ("pay_demo_108", "demo_cust_07", "insufficient_funds", 24999.0,
     "retry_24h", "retry_24h", "allowed", "ML recommendation passed all policy checks.",
     "failed", "evt_demo_108", "processed_with_errors", 0.71),
    ("pay_demo_109", "demo_cust_08", "technical_error", 999.0,
     "retry_2h", "retry_2h", "allowed", "ML recommendation passed all policy checks.",
     "executed", "evt_demo_109", "processed", 0.85),
    ("pay_demo_110", "demo_cust_09", "bank_decline", 75000.0,
     "payment_link", "escalation", "overridden", "High-value payment requires human approval.",
     "escalated", "evt_demo_110", "processed", 0.55),
    ("pay_demo_111", "demo_cust_10", "card_expired", 5499.0,
     "retry_2h", "payment_link", "overridden", "Expired card requires customer instrument update.",
     "executed", "evt_demo_111", "processed", 0.64),
    ("pay_demo_112", "demo_cust_11", "insufficient_funds", 12999.0,
     "payment_link", "stop", "blocked", "Retry limit reached; retry action is blocked.",
     "not_executed", "evt_demo_112", "processed", 0.60),
]

DIAGNOSIS = {
    "insufficient_funds": "customer_liquidity_issue",
    "technical_error": "transient_or_system_failure",
    "card_expired": "customer_instrument_needs_update",
    "mandate_revoked": "mandate_or_instrument_invalid",
    "bank_decline": "issuer_or_bank_decline",
    "disputed": "disputed_payment",
    "fraud_flag": "fraud_or_security_signal",
}


def demo_engine():
    from sqlalchemy import create_engine

    return create_engine(f"sqlite:///{DEMO_DB_PATH}")


def ensure_demo_data() -> Path:
    """Create + seed the demo DB if empty. Returns the DB path."""
    from sqlalchemy.orm import sessionmaker

    from src.db import Base
    from src.db_models import AuditLog, Customer, Payment, RecoveryAttempt, WebhookEvent

    engine = demo_engine()
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        from sqlalchemy import func, select

        if session.execute(select(func.count()).select_from(Payment)).scalar():
            return DEMO_DB_PATH
        now = datetime.utcnow()
        for i, (pid, cid, reason, amt, ml, final, pol, polreason, exe, eid, estat, conf) in enumerate(DEMO_ROWS):
            if session.get(Customer, cid) is None:
                session.add(Customer(customer_id=cid, segment="smb" if i % 3 == 0 else "individual",
                                     tenure_months=6 + i))
            ts = now - timedelta(hours=len(DEMO_ROWS) - i)
            session.add(Payment(payment_id=pid, customer_id=cid, order_id="order_" + pid,
                                amount=amt, currency="INR", decline_reason=reason, created_at=ts))
            session.add(RecoveryAttempt(payment_id=pid, ml_suggested_action=ml, final_action=final,
                                        policy_status=pol, policy_reason=polreason,
                                        execution_status=exe, created_at=ts))
            probs = {"retry_2h": 0.41, "retry_24h": 0.73, "payment_link": 0.54,
                     "escalation": 0.30, "stop": 0.0}
            session.add(AuditLog(
                event_id=eid, record_id="REC-" + pid, payment_id=pid,
                record={"record_id": "REC-" + pid, "event_id": eid, "payment_id": pid,
                        "amount": amt, "decline_reason": reason,
                        "diagnosis": {"decline_reason": reason,
                                      "classification": DIAGNOSIS.get(reason, "unknown_payment_failure")},
                        "ml_suggested_action": ml, "ml_confidence": conf,
                        "action_probabilities": probs,
                        "expected_net_recovery": {a: round(p * amt, 2) for a, p in probs.items()},
                        "policy_status": pol, "policy_reason": polreason,
                        "final_action": final, "execution_status": exe,
                        "execution_result": {"mode": "dry_run"} if exe == "executed" else {}},
                created_at=ts))
            session.add(WebhookEvent(event_id=eid,
                                     response={"status": estat, "payment_id": pid},
                                     created_at=ts))
        session.add(WebhookEvent(event_id="evt_demo_113",
                                 response={"status": "ignored", "reason": "Unsupported event: payment.captured"},
                                 created_at=now))
        session.commit()
        return DEMO_DB_PATH
    finally:
        session.close()
        engine.dispose()
