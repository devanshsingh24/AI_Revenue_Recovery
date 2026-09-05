from __future__ import annotations

from typing import Any, Dict

HARD_DECLINES = {"card_expired", "mandate_revoked", "disputed", "fraud_flag"}

# Named policy thresholds (values unchanged — names exist so display layers
# like the dashboard can import and show the live rules without hardcoding
# a second copy that could drift).
RETRY_LIMIT = 3
HIGH_VALUE_THRESHOLD = 50_000


def evaluate_policy(*, decline_reason: str, amount: float,
                     retry_count_so_far: int, ml_action: str) -> Dict[str, str]:
    """Single source of truth for guardrail rules. policy_validate_node
    (src/agent/nodes.py) and apply_guardrails() below both delegate here,
    so the rule set can't drift between the graph path and any standalone
    / pipeline caller."""
    reason = str(decline_reason or "unknown")
    retries = int(retry_count_so_far or 0)
    amt = float(amount or 0.0)

    if reason == "fraud_flag":
        return {"final_action": "stop", "status": "blocked",
                "reason": "Fraud/security signal: automatic recovery is prohibited."}
    if reason == "disputed":
        return {"final_action": "escalation", "status": "overridden",
                "reason": "Disputed payment requires human review."}
    if retries >= RETRY_LIMIT and ml_action in {"retry_2h", "retry_24h"}:
        return {"final_action": "payment_link", "status": "overridden",
                "reason": "Retry limit reached; retry action is blocked."}
    if amt > HIGH_VALUE_THRESHOLD and ml_action != "escalation":
        return {"final_action": "escalation", "status": "overridden",
                "reason": "High-value payment requires human approval."}
    if reason == "mandate_revoked" and ml_action in {"retry_2h", "retry_24h"}:
        return {"final_action": "payment_link", "status": "overridden",
                "reason": "Mandate revoked; bare retry is blocked."}
    if reason == "card_expired" and ml_action in {"retry_2h", "retry_24h"}:
        return {"final_action": "payment_link", "status": "overridden",
                "reason": "Expired card requires customer instrument update."}

    return {"final_action": ml_action, "status": "allowed",
            "reason": "ML recommendation passed all policy checks."}


def apply_guardrails(row: Dict[str, Any], ml_action: str) -> Dict[str, str]:
    """Row/dict entry point for non-graph callers (e.g. a pipeline script
    or notebook working off a flat record instead of LangGraph state)."""
    return evaluate_policy(
        decline_reason=row.get("decline_reason"),
        amount=row.get("amount"),
        retry_count_so_far=row.get("retry_count_so_far"),
        ml_action=ml_action,
    )