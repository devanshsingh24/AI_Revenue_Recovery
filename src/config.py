from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

_BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(_BASE_DIR / ".env")  # dev file relative to repo root, independent of CWD
load_dotenv()  # fallback: CWD / process environment (never overrides existing vars)


@dataclass(frozen=True)
class Settings:
    razorpay_key_id: str | None
    razorpay_key_secret: str | None
    razorpay_webhook_secret: str | None
    dry_run: bool
    database_url: str
    demo_database_url: str
    dashboard_api_base_url: str

    @classmethod
    def load(cls) -> "Settings":
        webhook_secret = os.getenv("RAZORPAY_WEBHOOK_SECRET")
        dry_run = os.getenv("RAZORPAY_DRY_RUN", "true").lower() == "true"
        key_id = os.getenv("RAZORPAY_KEY_ID")
        key_secret = os.getenv("RAZORPAY_KEY_SECRET")

        # Fail fast, at startup, not on first webhook: this is the config
        # validation layer H was missing. A misconfigured deployment should
        # refuse to start, not report healthy and fail three steps into a
        # live request.
        if not webhook_secret:
            raise RuntimeError(
                "RAZORPAY_WEBHOOK_SECRET is not set. Required even in dry-run "
                "mode — without it, no webhook can ever be verified."
            )
        if not dry_run and not (key_id and key_secret):
            raise RuntimeError(
                "RAZORPAY_DRY_RUN is false but RAZORPAY_KEY_ID/RAZORPAY_KEY_SECRET "
                "are missing. Set both, or set RAZORPAY_DRY_RUN=true for local dev."
            )

        return cls(
            razorpay_key_id=key_id, razorpay_key_secret=key_secret,
            razorpay_webhook_secret=webhook_secret, dry_run=dry_run,
            database_url=os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/revenue_recovery"),
            # Demo data lives in an actually separate database — never the
            # production connection with a flag. Defaults to a local file
            # next to the repo so a demo can never touch production rows.
            demo_database_url=os.getenv(
                "DEMO_DATABASE_URL",
                "sqlite:///./demo_revenue_recovery.db",
            ),
            dashboard_api_base_url=os.getenv(
                # API_BASE_URL is the deployment alias (Streamlit Cloud secrets /
                # Render env). DASHBOARD_API_BASE_URL keeps working for existing setups.
                "DASHBOARD_API_BASE_URL",
                os.getenv("API_BASE_URL", "http://127.0.0.1:8001"),
            ),
        )


settings = Settings.load()  # evaluated once at import time — this IS the fail-fast check