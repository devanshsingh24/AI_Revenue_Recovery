from __future__ import annotations
import hashlib, hmac, os
from typing import Any, Dict, Optional
import razorpay

class RazorpayClient:
    def __init__(self, key_id: Optional[str] = None, key_secret: Optional[str] = None,
                 webhook_secret: Optional[str] = None, dry_run: Optional[bool] = None):
        self.key_id = key_id or os.getenv("RAZORPAY_KEY_ID")
        self.key_secret = key_secret or os.getenv("RAZORPAY_KEY_SECRET")
        self.webhook_secret = webhook_secret or os.getenv("RAZORPAY_WEBHOOK_SECRET")
        if dry_run is None:
            dry_run = os.getenv("RAZORPAY_DRY_RUN", "true").lower() == "true"
        self.dry_run = dry_run
        self.client = None
        if self.key_id and self.key_secret:
            self.client = razorpay.Client(auth=(self.key_id, self.key_secret))

    def verify_webhook_signature(self, raw_body: bytes, received_signature: str) -> bool:
        if not self.webhook_secret:
            raise RuntimeError("RAZORPAY_WEBHOOK_SECRET is not configured.")
        expected = hmac.new(self.webhook_secret.encode(), raw_body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, received_signature)

    def create_order(self, amount_paise: int, currency: str, receipt: str,
                     notes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.dry_run:
            return {"mode": "dry_run", "operation": "create_order", "amount": amount_paise,
                    "currency": currency, "receipt": receipt, "notes": notes or {}}
        self._require_client()
        return self.client.order.create(data={"amount": amount_paise, "currency": currency,
                                               "receipt": receipt, "notes": notes or {}})

    def create_payment_link(self, amount_paise: int, currency: str, reference_id: str,
                            description: str) -> Dict[str, Any]:
        if self.dry_run:
            return {"mode": "dry_run", "operation": "create_payment_link", "amount": amount_paise,
                    "currency": currency, "reference_id": reference_id, "description": description}
        self._require_client()
        return self.client.payment_link.create({"amount": amount_paise, "currency": currency,
                                                 "reference_id": reference_id, "description": description,
                                                 "reminder_enable": True})

    def execute_recovery_action(self, action: str, amount: int, currency: str,
                                reference_id: str, description: str) -> Dict[str, Any]:
        if action == "retry_2h":
            return self.create_order(amount, currency, f"recovery-{reference_id}-2h", {"recovery_action": action})
        if action == "retry_24h":
            return self.create_order(amount, currency, f"recovery-{reference_id}-24h", {"recovery_action": action})
        if action == "payment_link":
            return self.create_payment_link(amount, currency, f"recovery-{reference_id}", description)
        raise ValueError(f"Unsupported executable action: {action}")

    def _require_client(self) -> None:
        if self.client is None:
            raise RuntimeError("Razorpay credentials are missing. Use RAZORPAY_DRY_RUN=true for local development.")
