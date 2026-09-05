"""Seed the DEMO database at scale (scripts/seed_demo_data.py).

Usage:
    python scripts/seed_demo_data.py [--n 1200]

Safety: refuses to run when DEMO_DATABASE_URL == DATABASE_URL (normalized
string compare) — demo rows can never land in production tables.

Representativeness: input contexts are sampled from data.py's own generated
dataset (same distributions the model trained on). Actions come from the REAL
trained model (score_all_actions) + REAL policy (evaluate_policy) + REAL
diagnosis (diagnose_node) — only the payment/customer inputs are synthetic.

Outcome simulation uses data.py's own ground-truth technique (action_logit
sigmoid per final action + binomial coin-flip) and is stored/displayed ONLY
under `simulated_*` names, which exist solely in the demo database.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

N_DEFAULT = 1200


def _normalize(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_DEFAULT)
    args = parser.parse_args(argv)

    import os

    demo_url = os.getenv("DEMO_DATABASE_URL", "")
    prod_url = os.getenv("DATABASE_URL", "")
    if not demo_url:
        print("DEMO_DATABASE_URL is not set — refusing to guess where demo data goes.",
              file=sys.stderr)
        return 2
    if _normalize(demo_url) == _normalize(prod_url):
        print("REFUSING: DEMO_DATABASE_URL == DATABASE_URL. Demo rows must never "
              "go into production tables.", file=sys.stderr)
        return 2

    import numpy as np
    import pandas as pd
    from sqlalchemy import MetaData, Table, create_engine
    from sqlalchemy.orm import sessionmaker

    print("Loading data.py generation (reuses its exact distributions)...")
    import data as gen  # noqa: deterministic seed=42; rewrites the identical CSV
    from data import action_logit

    from train_policy import ACTIONS, ACTION_COST, FEATURE_COLS, load_model
    from src.agent.nodes import diagnose_node
    from src.guardrails import evaluate_policy
    from src.db import Base
    from src import db_models as _db_models  # noqa: F401 - registers tables on Base before create_all

    model, _ = load_model()
    rng = np.random.default_rng(7)
    df = gen.df
    n = min(args.n, len(df))
    idx = rng.choice(len(df), size=n, replace=False)
    sample = df.iloc[idx].reset_index(drop=True)
    print(f"Sampled {n} input contexts from data.py output ({len(df)} rows).")

    demo_engine = create_engine(demo_url, pool_pre_ping=True)
    # Demo database is disposable: start clean so re-runs never duplicate PKs.
    Base.metadata.drop_all(bind=demo_engine)
    Base.metadata.create_all(bind=demo_engine)
    with demo_engine.begin() as conn:
        for col, typ in (("simulated_recovered", "BOOLEAN"),
                         ("simulated_recovered_amount", "FLOAT")):
            try:
                conn.exec_driver_sql(
                    f"ALTER TABLE recovery_attempts ADD COLUMN {col} {typ}")
            except Exception:
                pass  # already present on re-runs

    print("Scoring all records in one batched predict_proba call...")
    feat_cols = [c for c in FEATURE_COLS if c != "recovery_action"]
    big = pd.concat(
        [sample[feat_cols].assign(recovery_action=a) for a in ACTIONS],
        ignore_index=True)[FEATURE_COLS]
    proba_all = np.clip(model.predict_proba(big)[:, 1], 0.0, 1.0).reshape(len(ACTIONS), n).T
    amounts = sample["amount"].to_numpy(dtype=float)
    net_all = proba_all * amounts[:, None] - np.array([ACTION_COST[a] for a in ACTIONS])
    ml_idx = net_all.argmax(axis=1)

    print("Running real policy + diagnosis per record...")
    ml_actions, finals, pstats, preasons, diags = [], [], [], [], []
    for i, row in sample.iterrows():
        record = row.to_dict()
        ml_a = ACTIONS[int(ml_idx[i])]
        ml_actions.append(ml_a)
        policy = evaluate_policy(decline_reason=record.get("decline_reason"),
                                 amount=float(record.get("amount") or 0.0),
                                 retry_count_so_far=int(record.get("retry_count_so_far") or 0),
                                 ml_action=ml_a)
        finals.append(policy["final_action"])
        pstats.append(policy["status"])
        preasons.append(policy["reason"])
        diags.append(diagnose_node({"decline_reason": record.get("decline_reason")}).get("diagnosis", {}))
    finals = np.array(finals)

    print("Simulating outcomes (data.py coin-flip technique, vectorized)...")
    p_final = np.zeros(n)
    for a in ACTIONS:
        mask = finals == a
        if mask.any():
            p_final[mask] = 1.0 / (1.0 + np.exp(-action_logit(sample.loc[mask], a).to_numpy()))
    flips = rng.binomial(1, p_final).astype(bool)

    print("Bulk-inserting...")
    meta = MetaData()
    t_customers = Table("customers", meta, autoload_with=demo_engine)
    t_payments = Table("payments", meta, autoload_with=demo_engine)
    t_attempts = Table("recovery_attempts", meta, autoload_with=demo_engine)
    t_audits = Table("audit_logs", meta, autoload_with=demo_engine)
    t_events = Table("webhook_events", meta, autoload_with=demo_engine)
    now = datetime.utcnow()
    days = rng.uniform(0, 180, n)
    hours = rng.uniform(0, 24, n)

    seen, cust_rows = set(), []
    for _, row in sample.iterrows():
        cid = row["customer_id"]
        if cid not in seen:
            seen.add(cid)
            cust_rows.append({
                "customer_id": cid,
                "segment": row.get("customer_segment", "individual"),
                "tenure_months": int(row.get("customer_tenure_months") or 0),
                "past_payment_success_rate": float(row.get("past_payment_success_rate") or 0.5),
                "historical_engagement_score": float(row.get("historical_engagement_score") or 0.5),
                "previous_failed_payments": 0, "previous_recovered_payments": 0,
                "updated_at": now,
            })

    pay_rows, att_rows, aud_rows, evt_rows = [], [], [], []
    for i in range(n):
        pid, eid = f"pay_seed_{i:06d}", f"evt_seed_{i:06d}"
        ts = now - timedelta(days=float(days[i]), hours=float(hours[i]))
        exe = ("escalated" if finals[i] == "escalation"
               else "not_executed" if finals[i] == "stop" else "executed")
        probs = {a: float(proba_all[i][j]) for j, a in enumerate(ACTIONS)}
        sim_amt = round(float(amounts[i]), 2) if flips[i] else 0.0
        pay_rows.append({"payment_id": pid, "customer_id": sample.iloc[i]["customer_id"],
                         "order_id": "order_" + pid, "amount": float(amounts[i]),
                         "currency": "INR", "decline_reason": sample.iloc[i]["decline_reason"],
                         "created_at": ts})
        att_rows.append({"payment_id": pid, "ml_suggested_action": ml_actions[i],
                         "final_action": str(finals[i]), "policy_status": pstats[i],
                         "policy_reason": preasons[i], "execution_status": exe,
                         "created_at": ts, "simulated_recovered": bool(flips[i]),
                         "simulated_recovered_amount": sim_amt})
        aud_rows.append({"event_id": eid, "record_id": f"REC-SEED-{i:06d}", "payment_id": pid,
                         "record": {"record_id": f"REC-SEED-{i:06d}", "event_id": eid,
                                    "payment_id": pid, "amount": float(amounts[i]),
                                    "decline_reason": sample.iloc[i]["decline_reason"],
                                    "diagnosis": diags[i], "ml_suggested_action": ml_actions[i],
                                    "ml_confidence": probs[ml_actions[i]],
                                    "action_probabilities": probs,
                                    "expected_net_recovery": {a: float(net_all[i][j]) for j, a in enumerate(ACTIONS)},
                                    "policy_status": pstats[i], "policy_reason": preasons[i],
                                    "final_action": str(finals[i]), "execution_status": exe,
                                    "execution_result": {"mode": "dry_run"} if exe == "executed" else {}},
                         "created_at": ts})
        evt_rows.append({"event_id": eid, "response": {"status": "processed", "payment_id": pid},
                         "created_at": ts})
    evt_rows.append({"event_id": "evt_seed_ignored",
                     "response": {"status": "ignored", "reason": "Unsupported event: payment.captured"},
                     "created_at": now})

    Session = sessionmaker(bind=demo_engine)
    session = Session()
    try:
        for table, rows in ((t_customers, cust_rows), (t_payments, pay_rows),
                            (t_attempts, att_rows), (t_audits, aud_rows), (t_events, evt_rows)):
            for j in range(0, len(rows), 1000):
                session.execute(table.insert(), rows[j:j + 1000])
        session.commit()
    finally:
        session.close()
        demo_engine.dispose()
    rec_count = int(flips.sum())
    print(f"Seeded {n} demo rows (simulated recovered: {rec_count}, "
          f"rate {rec_count / n * 100:.1f}%). Production untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
