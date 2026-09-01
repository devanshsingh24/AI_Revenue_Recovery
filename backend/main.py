from __future__ import annotations
import os
from pathlib import Path
from typing import Any
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException, Request
from src.razorpay_client import RazorpayClient
from src.webhook_handler import WebhookHandler

FEATURE_COLS = [
    "amount", "decline_reason", "is_hard_decline", "payment_method", "customer_segment",
    "customer_tenure_months", "days_overdue", "retry_count_so_far",
    "past_payment_success_rate", "historical_engagement_score",
    "previous_failed_payments", "previous_recovered_payments", "recovery_action",
]
MODEL_PATH = os.getenv("MODEL_PATH", "models/recovery_catboost.cbm")
app = FastAPI(title="AI Revenue Recovery Agent", version="0.1.0")
webhook_handler: WebhookHandler | None = None

def load_model():
    path = Path(MODEL_PATH)
    if not path.exists():
        raise RuntimeError(f"Missing model: {path}. Train/export your CatBoost model and set MODEL_PATH.")
    model = CatBoostClassifier(); model.load_model(str(path)); return model

@app.on_event("startup")
def startup():
    global webhook_handler
    model = load_model()
    client = RazorpayClient()
    webhook_handler = WebhookHandler(model, FEATURE_COLS, client)

@app.get("/health")
def health():
    return {"status": "ok", "razorpay_mode": "dry_run" if os.getenv("RAZORPAY_DRY_RUN", "true").lower() == "true" else "test_mode"}

@app.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request):
    if webhook_handler is None:
        raise HTTPException(status_code=503, detail="Agent not initialized.")
    raw_body = await request.body()
    headers = {"x-razorpay-signature": request.headers.get("x-razorpay-signature", ""),
               "x-razorpay-event-id": request.headers.get("x-razorpay-event-id", "")}
    try:
        return webhook_handler.handle(raw_body, headers)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
