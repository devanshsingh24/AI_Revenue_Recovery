from __future__ import annotations
from src.guardrails import evaluate_policy
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from src.agent.state import RecoveryState

ACTIONS = ["retry_2h", "retry_24h", "payment_link", "escalation", "stop"]
ACTION_COST = {"retry_2h": 0.0, "retry_24h": 0.0, "payment_link": 20.0, "escalation": 500.0, "stop": 0.0}
DEFAULTS = {
    "customer_segment": "individual", "customer_tenure_months": 0, "days_overdue": 0,
    "retry_count_so_far": 0, "past_payment_success_rate": 0.5,
    "historical_engagement_score": 0.5, "previous_failed_payments": 0,
    "previous_recovered_payments": 0, "payment_method": None, "customer_id": None,
}


def diagnose_node(state: RecoveryState) -> RecoveryState:
    reason = str(state.get("decline_reason") or "unknown")
    classification = {
        "card_expired": "customer_instrument_needs_update",
        "mandate_revoked": "mandate_or_instrument_invalid",
        "disputed": "disputed_payment",
        "fraud_flag": "fraud_or_security_signal",
        "technical_error": "transient_or_system_failure",
        "insufficient_funds": "customer_liquidity_issue",
        "bank_decline": "issuer_or_bank_decline",
    }.get(reason, "unknown_payment_failure")
    is_hard_decline = reason in {"card_expired", "mandate_revoked", "disputed", "fraud_flag"}
    return {
        **state,
        "is_hard_decline": is_hard_decline,  # top-level: this is what predict_node/FEATURE_COLS read
        "diagnosis": {
            "decline_reason": reason,
            "classification": classification,
            "is_hard_decline": is_hard_decline,  # kept here too for anything reading diagnosis as a unit
        },
    }


def predict_node(state: RecoveryState, model: Any, feature_cols: List[str]) -> RecoveryState:
    if model is None:
        raise RuntimeError("CatBoost model is required for predict_node.")
    row_base: Dict[str, Any] = {}
    for col in feature_cols:
        if col == "recovery_action":
            continue
        value = state.get(col)
        row_base[col] = DEFAULTS.get(col) if value is None else value
    candidates = []
    for action in ACTIONS:
        row = dict(row_base)
        row["recovery_action"] = action
        candidates.append(row)
    X = pd.DataFrame(candidates)
    probabilities = np.clip(model.predict_proba(X)[:, 1], 0.0, 1.0)
    amount = float(state.get("amount") or 0.0)
    net_values = {a: float(p * amount - ACTION_COST[a]) for a, p in zip(ACTIONS, probabilities)}
    prob_map = {a: float(p) for a, p in zip(ACTIONS, probabilities)}
    suggested = max(net_values, key=net_values.get)
    return {**state, "ml_suggested_action": suggested,
            "action_probabilities": prob_map,
            "expected_net_recovery": net_values,
            "ml_confidence": prob_map[suggested]}


def decide_node(state: RecoveryState) -> RecoveryState:
    # Intentionally does not set final_action. Only policy_validate_node may do that.
    return {**state, "diagnosis": {
        **state.get("diagnosis", {}),
        "decision_basis": "maximize model-estimated expected net recovery subject to hard policy validation",
    }}



def policy_validate_node(state: RecoveryState) -> RecoveryState:
    """HARD GATE: only node allowed to set final_action."""
    ml_action = state.get("ml_suggested_action", "stop")
    result = evaluate_policy(
        decline_reason=state.get("decline_reason"),
        amount=state.get("amount"),
        retry_count_so_far=state.get("retry_count_so_far"),
        ml_action=ml_action,
    )
    updated = {
        **state,
        "final_action": result["final_action"],
        "policy_status": result["status"],
        "policy_reason": result["reason"],
    }
    if result["final_action"] == "stop":
        updated["execution_status"] = "not_executed"
    return updated

def execute_node(state: RecoveryState, razorpay_client: Any) -> RecoveryState:
    final_action = state.get("final_action", "stop")
    if final_action in {"stop", "escalation"}:
        return state
    result = razorpay_client.execute_recovery_action(
        action=final_action,
        amount=int(round(float(state.get("amount", 0.0)) * 100)),
        currency=state.get("currency", "INR"),
        reference_id=state.get("record_id") or state.get("payment_id") or "recovery",
        description=f"Revenue recovery for {state.get('payment_id', 'payment')}",
    )
    return {**state, "execution_status": "executed", "execution_result": result}


def escalate_node(state: RecoveryState) -> RecoveryState:
    if state.get("final_action") != "escalation":
        return state
    escalation = {"status": "queued", "reason": state.get("policy_reason", "Manual review required."),
                   "payment_id": state.get("payment_id"), "amount": state.get("amount")}
    return {**state, "escalation_result": escalation, "execution_status": "escalated"}


def audit_node(state: RecoveryState) -> RecoveryState:
    record = {
        "event_id": state.get("event_id"), "event_name": state.get("event_name"),
        "payment_id": state.get("payment_id"), "record_id": state.get("record_id"),
        "amount": state.get("amount"), "decline_reason": state.get("decline_reason"),
        "ml_suggested_action": state.get("ml_suggested_action"),
        "ml_confidence": state.get("ml_confidence"),
        "action_probabilities": state.get("action_probabilities", {}),
        "expected_net_recovery": state.get("expected_net_recovery", {}),
        "policy_status": state.get("policy_status"), "policy_reason": state.get("policy_reason"),
        "final_action": state.get("final_action"), "execution_status": state.get("execution_status"),
        "execution_result": state.get("execution_result", {}),
        "escalation_result": state.get("escalation_result", {}),
    }
    return {**state, "audit_record": record}
