from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class RecoveryState(TypedDict, total=False):
    event_id: str
    event_name: str
    payment_id: str
    order_id: Optional[str]
    record_id: Optional[str]
    amount: float
    currency: str
    decline_reason: str
    is_hard_decline: bool
    payment_method: Optional[str]
    customer_id: Optional[str]
    customer_segment: Optional[str]
    customer_tenure_months: int
    days_overdue: int
    retry_count_so_far: int
    past_payment_success_rate: float
    historical_engagement_score: float
    previous_failed_payments: int
    previous_recovered_payments: int
    diagnosis: Dict[str, Any]
    ml_suggested_action: str
    action_probabilities: Dict[str, float]
    expected_net_recovery: Dict[str, float]
    ml_confidence: float
    final_action: str
    policy_status: str
    policy_reason: str
    execution_status: str
    execution_result: Dict[str, Any]
    escalation_result: Dict[str, Any]
    audit_record: Dict[str, Any]
    errors: List[str]