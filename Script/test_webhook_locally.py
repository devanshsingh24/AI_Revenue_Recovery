from __future__ import annotations
import hashlib, hmac, json, os
import requests
WEBHOOK_URL = os.getenv("LOCAL_WEBHOOK_URL", "http://127.0.0.1:8000/webhooks/razorpay")
SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "dev-secret")
PAYLOAD = {
    "id": "evt_demo_001", "event": "payment.failed",
    "payload": {"payment": {"entity": {
        "id": "pay_demo_001", "order_id": "order_demo_001", "amount": 1800000, "currency": "INR", "method": "card",
        "error_reason": "insufficient_funds",
        "notes": {"record_id": "REC-DEMO-001", "customer_id": "CUST00001", "customer_segment": "individual",
                  "customer_tenure_months": 14, "days_overdue": 0, "retry_count_so_far": 0,
                  "past_payment_success_rate": 0.82, "historical_engagement_score": 0.64,
                  "previous_failed_payments": 1, "previous_recovered_payments": 2, "is_hard_decline": False}
    }}}
}
raw = json.dumps(PAYLOAD, separators=(",", ":")).encode()
signature = hmac.new(SECRET.encode(), raw, hashlib.sha256).hexdigest()
response = requests.post(WEBHOOK_URL, data=raw, headers={"Content-Type": "application/json",
    "X-Razorpay-Signature": signature, "X-Razorpay-Event-Id": PAYLOAD["id"]}, timeout=30)
print("Status:", response.status_code); print(response.text)
