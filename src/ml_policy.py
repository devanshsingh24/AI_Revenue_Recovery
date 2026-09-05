"""Deprecated compatibility shim.

The active scoring paths are ``train_policy.score_all_actions`` (standalone)
and ``src.agent.nodes.predict_node`` (graph). This module is kept only so any
legacy ``from src.ml_policy import ml_policy`` import keeps working; it
delegates to the same ``train_policy`` constants instead of redefining logic.
"""
from __future__ import annotations

from typing import Any, Dict


def ml_policy(row: Any, model: Any = None) -> Dict[str, Any]:
    """Score all actions with expected net recovery, legacy dict shape."""
    import numpy as np
    import pandas as pd

    from train_policy import ACTIONS, ACTION_COST, FEATURE_COLS, load_model

    if model is None:
        model, _ = load_model()
    values = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    base = {c: values.get(c) for c in FEATURE_COLS if c != "recovery_action"}
    candidates = pd.DataFrame([{**base, "recovery_action": a} for a in ACTIONS])
    probabilities = np.clip(model.predict_proba(candidates[FEATURE_COLS])[:, 1], 0.0, 1.0)
    amount = float(values.get("amount") or 0.0)
    net_ev = probabilities * amount - np.array([ACTION_COST[a] for a in ACTIONS])
    best_idx = int(net_ev.argmax())
    return {
        "suggested_action": ACTIONS[best_idx],
        "predicted_probability": float(probabilities[best_idx]),
        "expected_net_recovery": float(net_ev[best_idx]),
        "all_action_scores": {a: float(p) for a, p in zip(ACTIONS, probabilities)},
    }
