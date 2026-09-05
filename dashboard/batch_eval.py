"""Interactive batch evaluation for the dashboard (dashboard/batch_eval.py).

Demo-mode only: generates fresh synthetic failed-payment contexts using
data.py's distributions, scores them with the REAL model
(train_policy.load_model) + REAL policy (evaluate_policy), simulates
outcomes with data.py's coin-flip technique, and appends to the DEMO
database only. Never touches production, the agent, Razorpay, or schemas.

`run_batch` is a generator yielding (done_count, total) as chunks complete
so the Streamlit progress bar reflects actual processing.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Iterator, Tuple


def _normalize(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def run_batch(n: int) -> Iterator[Tuple[int, int, Dict[str, Any] | None]]:
    """Yield (done, total, summary|None); summary only on the final yield."""
    import numpy as np
    import pandas as pd
    from sqlalchemy import MetaData, Table, create_engine
    from sqlalchemy.orm import sessionmaker

    from src.config import settings
    if _normalize(settings.demo_database_url) == _normalize(settings.database_url):
        raise RuntimeError("REFUSING: DEMO_DATABASE_URL == DATABASE_URL.")
    if not settings.demo_database_url:
        raise RuntimeError("DEMO_DATABASE_URL is not set.")

    import data as gen  # noqa: reuses distributions/constants/logit only
    from train_policy import ACTIONS, ACTION_COST, FEATURE_COLS, load_model
    from src.agent.nodes import diagnose_node
    from src.guardrails import evaluate_policy
    from src.db import Base
    from src import db_models as _db_models  # noqa: F401 - registers tables

    started = time.time()
    rng = np.random.default_rng()
    model, _ = load_model()

    customers = pd.DataFrame({
        "customer_id": [f"batch_cust_{int(rng.integers(0, 1_000_000_000)):09d}_{i:03d}" for i in range(n)],
        "customer_segment": rng.choice(gen.segments, n, p=gen.segment_weights),
        "customer_tenure_months": np.round(rng.gamma(2.0, 8, n)).astype(int),
        "base_success_rate": rng.beta(6, 2, n),
        "base_engagement": rng.beta(2, 3, n),
    })
    ctx = pd.DataFrame({
        "amount": np.round(np.exp(rng.normal(8.2, 1.1, n)), 2),
        "decline_reason": rng.choice(gen.decline_reasons, n, p=gen.decline_weights),
        "payment_method": rng.choice(gen.payment_methods, n),
        "days_overdue": rng.poisson(6, n).astype(int),
        "retry_count_so_far": rng.poisson(1.0, n).astype(int),
        "previous_failed_payments": np.zeros(n, dtype=int),
        "previous_recovered_payments": np.zeros(n, dtype=int),
    })
    for col in ("customer_segment", "customer_tenure_months"):
        ctx[col] = customers[col].to_numpy()
    ctx["past_payment_success_rate"] = np.clip(
        customers["base_success_rate"].to_numpy() + rng.normal(0, 0.03, n), 0, 1)
    ctx["historical_engagement_score"] = np.clip(
        customers["base_engagement"].to_numpy() + rng.normal(0, 0.03, n), 0, 1)
    ctx["is_hard_decline"] = ctx["decline_reason"].isin(gen.hard_declines)

    feat = ctx[[c for c in FEATURE_COLS if c != "recovery_action"]]
    big = pd.concat([feat.assign(recovery_action=a) for a in ACTIONS],
                    ignore_index=True)[FEATURE_COLS]
    proba_all = np.clip(model.predict_proba(big)[:, 1], 0.0, 1.0).reshape(len(ACTIONS), n).T
    amounts = ctx["amount"].to_numpy(dtype=float)
    net_all = proba_all * amounts[:, None] - np.array([ACTION_COST[a] for a in ACTIONS])
    ml_idx = net_all.argmax(axis=1)

    finals, pstats, preasons, diags = [], [], [], []
    for i in range(n):
        ml_a = ACTIONS[int(ml_idx[i])]
        p = evaluate_policy(decline_reason=str(ctx["decline_reason"].iloc[i]),
                            amount=float(amounts[i]),
                            retry_count_so_far=int(ctx["retry_count_so_far"].iloc[i]),
                            ml_action=ml_a)
        finals.append(p["final_action"])
        pstats.append(p["status"])
        preasons.append(p["reason"])
        diags.append(diagnose_node({"decline_reason": str(ctx["decline_reason"].iloc[i])}).get("diagnosis", {}))
    finals_arr = np.array(finals)
    p_final = np.zeros(n)
    for a in ACTIONS:
        mask = finals_arr == a
        if mask.any():
            p_final[mask] = 1.0 / (1.0 + np.exp(-gen.action_logit(ctx.loc[mask], a).to_numpy()))
    flips = rng.binomial(1, p_final).astype(bool)

    demo_engine = create_engine(settings.demo_database_url, pool_pre_ping=True)
    Base.metadata.create_all(bind=demo_engine)
    with demo_engine.begin() as conn:
        insp_cols = {c["name"] for c in
                     __import__("sqlalchemy").inspect(demo_engine).get_columns("recovery_attempts")}
        for col, typ in (("simulated_recovered", "BOOLEAN"), ("simulated_recovered_amount", "FLOAT")):
            if col not in insp_cols:
                conn.exec_driver_sql(f"ALTER TABLE recovery_attempts ADD COLUMN {col} {typ}")
    meta = MetaData()
    t_customers = Table("customers", meta, autoload_with=demo_engine)
    t_payments = Table("payments", meta, autoload_with=demo_engine)
    t_attempts = Table("recovery_attempts", meta, autoload_with=demo_engine)
    t_audits = Table("audit_logs", meta, autoload_with=demo_engine)
    t_events = Table("webhook_events", meta, autoload_with=demo_engine)

    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f") + f"{int(rng.integers(0, 10000)):04d}"
    now = datetime.utcnow()
    Session = sessionmaker(bind=demo_engine)
    session = Session()
    final_counts: Dict[str, int] = {}
    policy_counts: Dict[str, int] = {}
    try:
        session.execute(t_customers.insert(), [
            {"customer_id": customers["customer_id"].iloc[i],
             "segment": str(customers["customer_segment"].iloc[i]),
             "tenure_months": int(customers["customer_tenure_months"].iloc[i]),
             "past_payment_success_rate": float(ctx["past_payment_success_rate"].iloc[i]),
             "historical_engagement_score": float(ctx["historical_engagement_score"].iloc[i]),
             "previous_failed_payments": 0, "previous_recovered_payments": 0,
             "updated_at": now} for i in range(n)])
        chunk = 25
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            pay_rows, att_rows, aud_rows, evt_rows = [], [], [], []
            for i in range(lo, hi):
                pid, eid = f"pay_batch_{stamp}_{i:04d}", f"evt_batch_{stamp}_{i:04d}"
                exe = ("escalated" if finals[i] == "escalation"
                       else "not_executed" if finals[i] == "stop" else "executed")
                probs = {a: float(proba_all[i][j]) for j, a in enumerate(ACTIONS)}
                pay_rows.append({"payment_id": pid, "customer_id": customers["customer_id"].iloc[i],
                                 "order_id": "order_" + pid, "amount": float(amounts[i]),
                                 "currency": "INR", "decline_reason": str(ctx["decline_reason"].iloc[i]),
                                 "created_at": now})
                att_rows.append({"payment_id": pid, "ml_suggested_action": ACTIONS[int(ml_idx[i])],
                                 "final_action": str(finals[i]), "policy_status": pstats[i],
                                 "policy_reason": preasons[i], "execution_status": exe,
                                 "created_at": now, "simulated_recovered": bool(flips[i]),
                                 "simulated_recovered_amount": round(float(amounts[i]), 2) if flips[i] else 0.0})
                aud_rows.append({"event_id": eid, "record_id": "REC-BATCH-" + pid, "payment_id": pid,
                                 "record": {"record_id": "REC-BATCH-" + pid, "event_id": eid,
                                            "payment_id": pid, "amount": float(amounts[i]),
                                            "decline_reason": str(ctx["decline_reason"].iloc[i]),
                                            "diagnosis": diags[i],
                                            "ml_suggested_action": ACTIONS[int(ml_idx[i])],
                                            "ml_confidence": probs[ACTIONS[int(ml_idx[i])]],
                                            "action_probabilities": probs,
                                            "expected_net_recovery": {a: float(net_all[i][j]) for j, a in enumerate(ACTIONS)},
                                            "policy_status": pstats[i], "policy_reason": preasons[i],
                                            "final_action": str(finals[i]), "execution_status": exe,
                                            "execution_result": {"mode": "dry_run"} if exe == "executed" else {}},
                                 "created_at": now})
                evt_rows.append({"event_id": eid, "response": {"status": "processed", "payment_id": pid},
                                 "created_at": now})
                final_counts[str(finals[i])] = final_counts.get(str(finals[i]), 0) + 1
                policy_counts[pstats[i]] = policy_counts.get(pstats[i], 0) + 1
            session.execute(t_payments.insert(), pay_rows)
            session.execute(t_attempts.insert(), att_rows)
            session.execute(t_audits.insert(), aud_rows)
            session.execute(t_events.insert(), evt_rows)
            session.commit()
            yield hi, n, None
    finally:
        session.close()
        demo_engine.dispose()
    yield n, n, {"processed": n, "final_action_counts": final_counts,
                 "policy_status_counts": policy_counts,
                 "simulated_recovered": int(flips.sum()),
                 "elapsed_s": round(time.time() - started, 1)}
