from __future__ import annotations
import argparse, json, os
from pathlib import Path
from catboost import CatBoostClassifier
from src.agent.graph import build_graph
from src.razorpay_client import RazorpayClient

FEATURE_COLS = [
    "amount", "decline_reason", "is_hard_decline", "payment_method", "customer_segment",
    "customer_tenure_months", "days_overdue", "retry_count_so_far",
    "past_payment_success_rate", "historical_engagement_score",
    "previous_failed_payments", "previous_recovered_payments", "recovery_action",
]

def load_model():
    path = Path(os.getenv("MODEL_PATH", "models/recovery_catboost.cbm"))
    if not path.exists():
        raise FileNotFoundError(f"CatBoost model not found at {path}. Set MODEL_PATH or export your model there.")
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model

def normalize_input(payload: dict) -> dict:
    entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
    notes = entity.get("notes") or {}
    return {
        "event_id": payload.get("event_id") or payload.get("id"), "event_name": payload.get("event", "payment.failed"),
        "payment_id": entity.get("id") or payload.get("payment_id"), "order_id": entity.get("order_id"),
        "record_id": notes.get("record_id") or entity.get("order_id") or entity.get("id"),
        "amount": float(entity.get("amount", payload.get("amount", 0))) / 100.0,
        "currency": entity.get("currency") or "INR",
        "decline_reason": entity.get("error_reason") or entity.get("error_description") or payload.get("decline_reason") or "unknown",
        "payment_method": entity.get("method"), "customer_id": notes.get("customer_id"),
        "customer_segment": notes.get("customer_segment", "individual"), "customer_tenure_months": int(notes.get("customer_tenure_months", 0)),
        "days_overdue": int(notes.get("days_overdue", 0)), "retry_count_so_far": int(notes.get("retry_count_so_far", 0)),
        "past_payment_success_rate": float(notes.get("past_payment_success_rate", 0.5)),
        "historical_engagement_score": float(notes.get("historical_engagement_score", 0.5)),
        "previous_failed_payments": int(notes.get("previous_failed_payments", 0)),
        "previous_recovered_payments": int(notes.get("previous_recovered_payments", 0)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()
    with open(args.payload, "r", encoding="utf-8") as f:
        payload = json.load(f)
    model = load_model()
    client = RazorpayClient()
    graph = build_graph(model, FEATURE_COLS, client)
    result = graph.invoke(normalize_input(payload))
    print(json.dumps({"ml_suggested_action": result.get("ml_suggested_action"),
                      "policy_status": result.get("policy_status"), "policy_reason": result.get("policy_reason"),
                      "final_action": result.get("final_action"), "execution_status": result.get("execution_status"),
                      "audit_record": result.get("audit_record")}, indent=2, default=str))

if __name__ == "__main__":
    main()
