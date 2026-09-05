"""HTTP client for the dashboard API (dashboard/api_client.py).

Keeps raw HTTP out of the Streamlit pages. Base URL comes from
DASHBOARD_API_BASE_URL (same Settings object — no second config path).
Raises RuntimeError with a human message when the API is unreachable.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

try:
    from src.config import settings

    _DEFAULT_BASE = settings.dashboard_api_base_url
except Exception:
    _DEFAULT_BASE = "http://127.0.0.1:8001"

TIMEOUT = 15


def base_url() -> str:
    import os

    # API_BASE_URL is the deployment alias (Streamlit Cloud secrets).
    # DASHBOARD_API_BASE_URL keeps working for existing local setups.
    return os.getenv(
        "DASHBOARD_API_BASE_URL", os.getenv("API_BASE_URL", _DEFAULT_BASE)
    ).rstrip("/")


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    try:
        response = requests.get(f"{base_url()}{path}", params=params or {}, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Dashboard API unreachable at {base_url()}. "
            f"Start it with: uvicorn backend.dashboard_api:dashboard_app --port 8001 "
            f"({type(exc).__name__})")


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.date().isoformat() if dt else None


def get_kpis(start: Optional[datetime] = None, end: Optional[datetime] = None,
             demo: bool = False) -> Dict[str, Any]:
    return _get("/api/kpis", {"start": _iso(start), "end": _iso(end), "demo": demo})


def get_actions(start: Optional[datetime] = None, end: Optional[datetime] = None,
                demo: bool = False) -> Dict[str, Any]:
    return _get("/api/actions-breakdown", {"start": _iso(start), "end": _iso(end), "demo": demo})


def get_decline_reasons(start: Optional[datetime] = None, end: Optional[datetime] = None,
                        demo: bool = False) -> Dict[str, Any]:
    return _get("/api/decline-reasons", {"start": _iso(start), "end": _iso(end), "demo": demo})


def get_policy(start: Optional[datetime] = None, end: Optional[datetime] = None,
               demo: bool = False) -> Dict[str, Any]:
    return _get("/api/policy-breakdown", {"start": _iso(start), "end": _iso(end), "demo": demo})


def get_detail_rows(start: Optional[datetime] = None, end: Optional[datetime] = None,
                    demo: bool = False, limit: int = 5000) -> List[Dict[str, Any]]:
    return _get("/api/recovery-detail",
                {"start": _iso(start), "end": _iso(end), "demo": demo, "limit": limit})["rows"]


def get_payment(payment_id: str, demo: bool = False) -> Dict[str, Any]:
    return _get(f"/api/payment/{payment_id}", {"demo": demo})


def get_trail(payment_id: Optional[str] = None, customer_id: Optional[str] = None,
              demo: bool = False) -> List[Dict[str, Any]]:
    return _get("/api/audit-trail",
                {"payment_id": payment_id, "customer_id": customer_id,
                 "demo": demo})["entries"]


def get_events(limit: int = 100, demo: bool = False) -> List[Dict[str, Any]]:
    return _get("/api/webhook-events", {"limit": limit, "demo": demo})["events"]


def get_model_info() -> Dict[str, Any]:
    return _get("/api/model-info")
