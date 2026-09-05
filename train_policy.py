from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "recovery_full_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "recovery_catboost.cbm"
FEATURE_CONFIG_PATH = MODEL_DIR / "feature_config.json"

ACTIONS = ["retry_2h", "retry_24h", "payment_link", "escalation", "stop"]
ACTION_COST = {"retry_2h": 0.0, "retry_24h": 0.0, "payment_link": 20.0, "escalation": 500.0, "stop": 0.0}

# Single source of truth for the feature schema. src/agent/nodes.py's predict_node
# takes feature_cols as an argument specifically so this list is defined once, here,
# and passed through rather than re-declared at each call site.
FEATURE_COLS = [
    "amount", "decline_reason", "is_hard_decline", "payment_method", "customer_segment",
    "customer_tenure_months", "days_overdue", "retry_count_so_far",
    "past_payment_success_rate", "historical_engagement_score",
    "previous_failed_payments", "previous_recovered_payments", "recovery_action",
]
CAT_FEATURES = ["decline_reason", "payment_method", "customer_segment", "recovery_action"]


def train(data_path: Path = DATA_PATH) -> Tuple[CatBoostClassifier, float]:
    """Train the recovery-action classifier. Does not save anything by itself."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    df = pd.read_csv(data_path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X, y = df[FEATURE_COLS], df["recovered"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = CatBoostClassifier(
        iterations=400, learning_rate=0.05, depth=6,
        cat_features=CAT_FEATURES, verbose=False, random_state=42,
    )
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, auc


def save(model: CatBoostClassifier, held_out_roc_auc: float | None = None,
         trained_at: str | None = None) -> None:
    """Persist model + feature schema. When training metrics are passed (the
    __main__ path does), they are stored alongside — never fabricated."""
    from datetime import datetime, timezone

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    config: Dict[str, Any] = {"feature_cols": FEATURE_COLS, "cat_features": CAT_FEATURES}
    if held_out_roc_auc is not None:
        config["held_out_roc_auc"] = round(float(held_out_roc_auc), 3)
        config["trained_at"] = trained_at or datetime.now(timezone.utc).isoformat()
    FEATURE_CONFIG_PATH.write_text(json.dumps(config, indent=2))


def load_model() -> Tuple[CatBoostClassifier, List[str]]:
    """For inference call sites (graph.py, backend/main.py, run_agent.py).
    Returns (model, feature_cols) — pass feature_cols straight into predict_node.
    Raises if train_policy.py has never been run with __main__."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run `python train_policy.py` first."
        )
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    config = json.loads(FEATURE_CONFIG_PATH.read_text())
    return model, config["feature_cols"]


def score_all_actions(model: CatBoostClassifier, row: Dict[str, Any]) -> str:
    """Standalone convenience wrapper mirroring predict_node's logic, for CLI/debug use."""
    base = {c: row.get(c) for c in FEATURE_COLS if c != "recovery_action"}
    candidates = pd.DataFrame([{**base, "recovery_action": a} for a in ACTIONS])
    proba = np.clip(model.predict_proba(candidates[FEATURE_COLS])[:, 1], 0.0, 1.0)
    amount = float(row.get("amount") or 0.0)
    net_ev = proba * amount - np.array([ACTION_COST[a] for a in ACTIONS])
    return ACTIONS[int(net_ev.argmax())]


if __name__ == "__main__":
    trained_model, held_out_auc = train()
    print("Held-out ROC-AUC:", round(held_out_auc, 3))
    save(trained_model, held_out_roc_auc=held_out_auc)
    print("Saved model to:", MODEL_PATH)
    print("Saved feature config to:", FEATURE_CONFIG_PATH)