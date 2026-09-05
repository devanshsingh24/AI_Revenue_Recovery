"""Batch evaluation over the existing synthetic dataset.

Reuses the tested layers (``train_policy.score_all_actions`` +
``src.guardrails.apply_guardrails``) instead of redefining policy logic.
Writes an audit CSV under ``logs/``. No live Razorpay execution.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "data" / "recovery_full_dataset.csv"
DEFAULT_OUT = BASE_DIR / "logs" / "recovery_audit_log.csv"


def run_batch(data_path: Path = DEFAULT_DATA, limit: int | None = None) -> dict:
    from train_policy import load_model, score_all_actions
    from src.guardrails import apply_guardrails

    model, _ = load_model()
    df = pd.read_csv(data_path)
    if limit is not None:
        df = df.head(limit)

    audit_records = []
    override_count = 0
    blocked_count = 0
    for _, row in df.iterrows():
        record = row.to_dict()
        ml_action = score_all_actions(model, record)
        policy_result = apply_guardrails(record, ml_action)
        if policy_result["status"] == "overridden":
            override_count += 1
        elif policy_result["status"] == "blocked":
            blocked_count += 1
        audit_records.append({
            "record_id": record.get("record_id"),
            "ml_suggested_action": ml_action,
            "final_action": policy_result["final_action"],
            "policy_status": policy_result["status"],
            "policy_reason": policy_result["reason"],
        })
    return {
        "records": audit_records,
        "override_count": override_count,
        "blocked_count": blocked_count,
        "total": len(audit_records),
    }


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    summary = run_batch(Path(args.data), args.limit)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary["records"]).to_csv(out_path, index=False)
    print(f"Evaluated {summary['total']} records "
          f"(overridden={summary['override_count']}, blocked={summary['blocked_count']}). "
          f"Wrote audit log to: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
