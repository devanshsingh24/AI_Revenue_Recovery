from __future__ import annotations
import json
from typing import Any, Dict
from src.agent.graph import build_graph
from src.razorpay_client import RazorpayClient

class WebhookHandler:
    def __init__(self, model: Any, feature_cols: list[str], razorpay_client: RazorpayClient):
        self.razorpay_client = razorpay_client
        self.graph = build_graph(model, feature_cols, razorpay_client)

    def handle(self, raw_body: bytes, headers: Dict[str, str]) -> Dict[str, Any]:
        signature = headers.get("x-razorpay-signature")
        if not signature:
            raise ValueError("Missing X-Razorpay-Signature header.")
        if not self.razorpay_client.verify_webhook_signature(raw_body, signature):
            raise ValueError("Invalid Razorpay webhook signature.")
        payload = json.loads(raw_body.decode("utf-8"))
        if payload.get("event") != "payment.failed":
            return {"status": "ignored", "reason": f"Unsupported event: {payload.get('event')}"}
        result = self.graph.invoke(self._normalize_payment_failed(payload))
        return {"status": "processed", "payment_id": result.get("payment_id"),
                "ml_suggested_action": result.get("ml_suggested_action"),
                "policy_status": result.get("policy_status"), "policy_reason": result.get("policy_reason"),
                "final_action": result.get("final_action"), "execution_status": result.get("execution_status"),
                "audit_record": result.get("audit_record")}

    @staticmethod
    def _normalize_payment_failed(payload: Dict[str, Any]) -> Dict[str, Any]:
        entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
        notes = entity.get("notes") or {}
        return {
            "event_id": payload.get("event_id") or payload.get("id"),
            "event_name": payload.get("event", "payment.failed"),
            "payment_id": entity.get("id"), "order_id": entity.get("order_id"),
            "record_id": notes.get("record_id") or entity.get("order_id") or entity.get("id"),
            "amount": float(entity.get("amount", 0)) / 100.0, "currency": entity.get("currency", "INR"),
            "decline_reason": entity.get("error_reason") or entity.get("error_description") or "unknown",
            "payment_method": entity.get("method"), "customer_id": notes.get("customer_id"),
            "customer_segment": notes.get("customer_segment", "individual"),
            "customer_tenure_months": int(notes.get("customer_tenure_months", 0)),
            "days_overdue": int(notes.get("days_overdue", 0)),
            "retry_count_so_far": int(notes.get("retry_count_so_far", 0)),
            "past_payment_success_rate": float(notes.get("past_payment_success_rate", 0.5)),
            "historical_engagement_score": float(notes.get("historical_engagement_score", 0.5)),
            "previous_failed_payments": int(notes.get("previous_failed_payments", 0)),
            "previous_recovered_payments": int(notes.get("previous_recovered_payments", 0)),
            "is_hard_decline": bool(notes.get("is_hard_decline", False)),
        }
