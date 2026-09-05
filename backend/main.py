from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException, Request
from src.agent.graph import build_graph_from_trained_model
from src.razorpay_client import RazorpayClient
from src.webhook_handler import WebhookHandler
from src.db import init_db, SessionLocal
from src.repository import PostgresIdempotencyStore

app = FastAPI(title="AI Revenue Recovery Agent", version="0.1.0")
webhook_handler: WebhookHandler | None = None


@app.on_event("startup")
def startup():
    global webhook_handler
    init_db()  # creates tables if missing — see db.py docstring for the migrations caveat
    client = RazorpayClient()
    graph = build_graph_from_trained_model(client)
    idempotency_store = PostgresIdempotencyStore(SessionLocal)
    webhook_handler = WebhookHandler(graph, client, idempotency_store)


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