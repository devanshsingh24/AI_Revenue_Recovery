from __future__ import annotations
from typing import Any, Dict
from src.db import get_session
from src import repository


def persist_audit_record(result: Dict[str, Any]) -> None:
    """Replaces the broken top-level snippet that referenced undefined
    row/ml_policy/apply_guardrails at import time. Called once, after a
    graph run completes, with the full result dict run_recovery() returns.
    Writes the audit record AND the recovery attempt AND updates customer
    history in one transaction — get_session() commits on success, rolls
    back all three together on failure, so audit logs can't end up
    recorded while the recovery-attempt/customer update silently didn't."""
    with get_session() as session:
        repository.record_recovery_attempt(session, result)
        repository.write_audit_log(session, result)