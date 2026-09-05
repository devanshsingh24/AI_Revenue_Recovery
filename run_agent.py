from __future__ import annotations
import argparse, json
from src.agent.graph import build_graph_from_trained_model, run_recovery
from src.razorpay_client import RazorpayClient
from src.audit import persist_audit_record

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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    args = parser.parse_args(argv)
    with open(args.payload, "r", encoding="utf-8") as f:
        payload = json.load(f)

    client = RazorpayClient()
    graph = build_graph_from_trained_model(client)  # loads model + feature_cols from train_policy.py, one place
    result = run_recovery(graph, normalize_input(payload))  # error-contained: always returns an audit_record

    # CLI path must persist too — webhook_handler isn't the only entry point.
    # Best-effort: never crash the CLI when the database is unavailable locally.
    persist_error = None
    try:
        persist_audit_record(result)
    except Exception as exc:  # noqa: BLE001 - surface, don't fail the CLI
        persist_error = f"{type(exc).__name__}: {exc}"

    print(json.dumps({"ml_suggested_action": result.get("ml_suggested_action"),
                      "policy_status": result.get("policy_status"), "policy_reason": result.get("policy_reason"),
                      "final_action": result.get("final_action"), "execution_status": result.get("execution_status"),
                      "errors": result.get("errors"),
                      "persist_error": persist_error,
                      "audit_record": result.get("audit_record")}, indent=2, default=str))


if __name__ == "__main__":
    main()