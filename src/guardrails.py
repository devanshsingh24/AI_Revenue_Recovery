def apply_guardrails(row, ml_action):

    reason = row["decline_reason"]
    retries = row["retry_count_so_far"]
    amount = row["amount"]

    # Rule 1: Fraud must never be automatically retried
    if reason == "fraud_flag":
        return {
            "final_action": "stop",
            "status": "blocked",
            "reason": "Fraud flag: automatic recovery prohibited"
        }

    # Rule 2: Disputed payment requires human review
    if reason == "disputed":
        return {
            "final_action": "escalation",
            "status": "overridden",
            "reason": "Disputed payment requires manual review"
        }

    # Rule 3: Too many retries
    if retries >= 3 and ml_action in ["retry_2h", "retry_24h"]:
        return {
            "final_action": "payment_link",
            "status": "overridden",
            "reason": "Maximum retry limit reached"
        }

    # Rule 4: High-value payment
    if amount > 50000 and ml_action != "escalation":
        return {
            "final_action": "escalation",
            "status": "overridden",
            "reason": "High-value payment requires human approval"
        }

    # Otherwise ML recommendation is allowed
    return {
        "final_action": ml_action,
        "status": "allowed",
        "reason": "ML action passed all policy checks"
    }