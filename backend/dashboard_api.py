"""Read-only dashboard HTTP API (backend/dashboard_api.py).

Standalone app (`uvicorn backend.dashboard_api:dashboard_app --port 8001`),
deliberately NOT mounted into the webhook app so the two can be separated
later. READ-ONLY: every route is GET; no route touches the agent, Razorpay,
or any write path — all work delegates to src/dashboard_queries.py.

`demo=true` routes to the physically separate demo database
(DEMO_DATABASE_URL). Simulated recovery fields appear ONLY in demo
responses and are absent (not zeroed) in production responses.
"""
from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict, Optional

from fastapi import FastAPI, Query
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from src import dashboard_queries as dq
from src.config import settings

dashboard_app = FastAPI(title="Revenue Recovery Dashboard API", version="0.1.0")

_prod_engine = create_engine(settings.database_url, pool_pre_ping=True)
_demo_engine = create_engine(settings.demo_database_url, pool_pre_ping=True)
_ProdSession = sessionmaker(bind=_prod_engine, autoflush=False, autocommit=False)
_DemoSession = sessionmaker(bind=_demo_engine, autoflush=False, autocommit=False)


def _session(demo: bool):
    return (_DemoSession if demo else _ProdSession)()


def _parse_range(start: Optional[str], end: Optional[str]):
    parsed_start = parsed_end = None
    if start:
        parsed_start = datetime.combine(datetime.fromisoformat(start).date(), time.min)
    if end:
        parsed_end = datetime.combine(datetime.fromisoformat(end).date(), time.max)
    return parsed_start, parsed_end


def _demo_simulated() -> Dict[str, Any]:
    """Simulated recovery figures from the DEMO database only.

    Reads the demo-only `simulated_*` columns (which do not exist in the
    production schema) via reflected Core — production can never serve these.
    """
    from sqlalchemy import MetaData, Table

    meta = MetaData()
    attempts = Table("recovery_attempts", meta, autoload_with=_demo_engine)
    session = _DemoSession()
    try:
        total = session.execute(select(func.count()).select_from(attempts)).scalar() or 0
        rec = session.execute(
            select(func.count()).select_from(attempts)
            .where(attempts.c.simulated_recovered.is_(True))).scalar() or 0
        amount = session.execute(
            select(func.sum(attempts.c.simulated_recovered_amount))).scalar() or 0.0
        return {
            "simulated_recovery_rate": (rec / total * 100.0) if total else 0.0,
            "simulated_recovered_count": int(rec),
            "simulated_recovered_amount": float(amount),
        }
    finally:
        session.close()


@dashboard_app.get("/api/kpis")
def kpis(start: Optional[str] = None, end: Optional[str] = None,
         demo: bool = False) -> Dict[str, Any]:
    s, e = _parse_range(start, end)
    session = _session(demo)
    try:
        total = dq.get_total_failed_payments(session, s, e)
        final = dq.get_final_action_counts(session, s, e)
        policy = dq.get_policy_status_counts(session, s, e)
        execution = dq.get_execution_status_counts(session, s, e)
        decline = dq.get_decline_reason_breakdown(session, s, e)
        amounts = dq.get_amount_attempted_by_final_action(session, s, e)
        actions = sum(v for k, v in final.items() if k in ("retry_2h", "retry_24h", "payment_link"))
        out: Dict[str, Any] = {
            "total_failed": total,
            "actions_taken": actions,
            "action_rate_pct": (actions / total * 100.0) if total else 0.0,
            "escalated_pct": (final.get("escalation", 0) / total * 100.0) if total else 0.0,
            "stopped_pct": (final.get("stop", 0) / total * 100.0) if total else 0.0,
            "final_action_counts": final,
            "policy_status_counts": policy,
            "execution_status_counts": execution,
            "decline_reason_counts": decline,
            "amount_attempted_by_final_action": amounts,
        }
        if demo:
            out.update(_demo_simulated())
        return out
    finally:
        session.close()


@dashboard_app.get("/api/actions-breakdown")
def actions_breakdown(start: Optional[str] = None, end: Optional[str] = None,
                       demo: bool = False) -> Dict[str, Any]:
    s, e = _parse_range(start, end)
    session = _session(demo)
    try:
        return {
            "final_action_counts": dq.get_final_action_counts(session, s, e),
            "execution_status_counts": dq.get_execution_status_counts(session, s, e),
            "amount_attempted_by_final_action": dq.get_amount_attempted_by_final_action(session, s, e),
        }
    finally:
        session.close()


@dashboard_app.get("/api/decline-reasons")
def decline_reasons(start: Optional[str] = None, end: Optional[str] = None,
                     demo: bool = False) -> Dict[str, Any]:
    s, e = _parse_range(start, end)
    session = _session(demo)
    try:
        return {"decline_reason_counts": dq.get_decline_reason_breakdown(session, s, e)}
    finally:
        session.close()


@dashboard_app.get("/api/policy-breakdown")
def policy_breakdown(start: Optional[str] = None, end: Optional[str] = None,
                      demo: bool = False) -> Dict[str, Any]:
    s, e = _parse_range(start, end)
    session = _session(demo)
    try:
        return {
            "policy_status_counts": dq.get_policy_status_counts(session, s, e),
            "model_stats": dq.get_model_confidence_stats(session, s, e),
        }
    finally:
        session.close()


@dashboard_app.get("/api/recovery-detail")
def recovery_detail(start: Optional[str] = None, end: Optional[str] = None,
                    demo: bool = False, limit: int = Query(default=500, le=5000),
                    offset: int = Query(default=0, ge=0)) -> Dict[str, Any]:
    s, e = _parse_range(start, end)
    session = _session(demo)
    try:
        rows = dq.get_recovery_detail_rows(session, s, e, limit=limit + offset)
        return {"rows": rows[offset:offset + limit], "limit": limit, "offset": offset}
    finally:
        session.close()


@dashboard_app.get("/api/payment/{payment_id}")
def payment_detail(payment_id: str, demo: bool = False) -> Dict[str, Any]:
    session = _session(demo)
    try:
        detail = dq.get_payment_detail(session, payment_id)
        trail = dq.get_audit_trail(session, payment_id=payment_id) if detail else []
        return {"detail": detail, "trail": trail[0]["steps"] if trail else []}
    finally:
        session.close()


@dashboard_app.get("/api/audit-trail")
def audit_trail(payment_id: Optional[str] = None, customer_id: Optional[str] = None,
                demo: bool = False, limit: int = Query(default=50, le=200)) -> Dict[str, Any]:
    session = _session(demo)
    try:
        entries = dq.get_audit_trail(session, payment_id=payment_id,
                                     customer_id=customer_id, limit=limit)
        return {"entries": [
            {"payment_id": e["payment_id"], "steps": e["steps"], "detail": e.get("detail") or {}}
            for e in entries]}
    finally:
        session.close()


@dashboard_app.get("/api/webhook-events")
def webhook_events(limit: int = Query(default=50, le=500), demo: bool = False) -> Dict[str, Any]:
    session = _session(demo)
    try:
        return {"events": dq.get_recent_webhook_events(session, limit)}
    finally:
        session.close()


def _read_model_info(config_path=None) -> Dict[str, Any]:
    """Real training metrics from feature_config.json, or available=False.

    Old artifacts predating the held_out_roc_auc key yield no number —
    never a fabricated placeholder.
    """
    import json
    from pathlib import Path

    from train_policy import FEATURE_CONFIG_PATH

    path = Path(config_path) if config_path else FEATURE_CONFIG_PATH
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"available": False}
    if "held_out_roc_auc" not in config:
        return {"available": False}
    return {"available": True, "held_out_roc_auc": config["held_out_roc_auc"],
            "trained_at": config.get("trained_at")}


@dashboard_app.get("/api/model-info")
def model_info() -> Dict[str, Any]:
    return _read_model_info()


@dashboard_app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "health": "/health", "docs": "/docs"}


@dashboard_app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
