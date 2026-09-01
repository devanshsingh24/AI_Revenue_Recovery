ml_action = ml_policy(row)

policy_result = apply_guardrails(row, ml_action)

audit_record = {
    "record_id": row["record_id"],
    "ml_suggested_action": ml_action,
    "final_action": policy_result["final_action"],
    "policy_status": policy_result["status"],
    "policy_reason": policy_result["reason"]
}


