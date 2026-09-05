from __future__ import annotations
import json
from typing import Any, Dict
from src.agent.graph import run_recovery
from src.razorpay_client import RazorpayClient
from src.audit import persist_audit_record
from src.db import get_session
from src import repository


class WebhookHandler:
    def __init__(self, compiled_graph: Any, razorpay_client: RazorpayClient,
                 idempotency_store: repository.PostgresIdempotencyStore):
        self.graph = compiled_graph
        self.razorpay_client = razorpay_client
        self.idempotency_store = idempotency_store

    def handle(self, raw_body: bytes, headers: Dict[str, str]) -> Dict[str, Any]:
        signature = headers.get("x-razorpay-signature")
        if not signature:
            raise ValueError("Missing X-Razorpay-Signature header.")
        if not self.razorpay_client.verify_webhook_signature(raw_body, signature):
            raise ValueError("Invalid Razorpay webhook signature.")

        payload = json.loads(raw_body.decode("utf-8"))
        event_id = payload.get("event_id") or payload.get("id") or headers.get("x-razorpay-event-id")

        if event_id:
            cached = self.idempotency_store.get(event_id)
            if cached is not None:
                return {**cached, "status": "duplicate_ignored"}

        if payload.get("event") != "payment.failed":
            return {"status": "ignored", "reason": f"Unsupported event: {payload.get('event')}"}

        state = self._normalize_payment_failed(payload)
        result = run_recovery(self.graph, state)
        result["event_id"] = event_id  # carried through so persist_audit_record can use it

        # Persistence: record the payment first, then the recovery attempt +
        # audit log via persist_audit_record (two transactions; see Step 8 tests).
        with get_session() as session:
            repository.record_payment(
                session, payment_id=result.get("payment_id"), customer_id=state.get("customer_id"),
                order_id=state.get("order_id"), amount=state.get("amount", 0.0),
                currency=state.get("currency", "INR"), decline_reason=state.get("decline_reason"),
            )
        persist_audit_record(result)

        response = {"status": "processed" if not result.get("errors") else "processed_with_errors",
                    "payment_id": result.get("payment_id"),
                    "ml_suggested_action": result.get("ml_suggested_action"),
                    "policy_status": result.get("policy_status"), "policy_reason": result.get("policy_reason"),
                    "final_action": result.get("final_action"), "execution_status": result.get("execution_status"),
                    "errors": result.get("errors")}

        if event_id:
            self.idempotency_store.set(event_id, response)

        return response

    @staticmethod
    def _normalize_payment_failed(payload: Dict[str, Any]) -> Dict[str, Any]:
        entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
        notes = entity.get("notes") or {}
        customer_id = notes.get("customer_id")

        # DB-first feature lookup, per your critique: notes is only the
        # fallback for a customer_id we've never recorded before.
        db_features: Dict[str, Any] = {}
        if customer_id:
            with get_session() as session:
                repository.upsert_customer_from_notes(session, customer_id, notes)
                db_features = repository.get_customer_features(session, customer_id) or {}

        return {
            "event_id": payload.get("event_id") or payload.get("id"),
            "event_name": payload.get("event", "payment.failed"),
            "payment_id": entity.get("id"), "order_id": entity.get("order_id"),
            "record_id": notes.get("record_id") or entity.get("order_id") or entity.get("id"),
            "amount": float(entity.get("amount", 0)) / 100.0, "currency": entity.get("currency", "INR"),
            "decline_reason": entity.get("error_reason") or entity.get("error_description") or "unknown",
            "payment_method": entity.get("method"), "customer_id": customer_id,
            "customer_segment": db_features.get("customer_segment", notes.get("customer_segment", "individual")),
            "customer_tenure_months": db_features.get("customer_tenure_months", int(notes.get("customer_tenure_months", 0))),
            "days_overdue": int(notes.get("days_overdue", 0)),
            "retry_count_so_far": int(notes.get("retry_count_so_far", 0)),
            "past_payment_success_rate": db_features.get("past_payment_success_rate", float(notes.get("past_payment_success_rate", 0.5))),
            "historical_engagement_score": db_features.get("historical_engagement_score", float(notes.get("historical_engagement_score", 0.5))),
            "previous_failed_payments": db_features.get("previous_failed_payments", int(notes.get("previous_failed_payments", 0))),
            "previous_recovered_payments": db_features.get("previous_recovered_payments", int(notes.get("previous_recovered_payments", 0))),
        }