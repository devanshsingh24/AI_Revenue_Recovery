import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

decline_reasons = ["insufficient_funds", "card_expired", "bank_decline",
                    "technical_error", "mandate_revoked", "disputed", "fraud_flag"]
decline_weights = [0.35, 0.15, 0.20, 0.10, 0.10, 0.07, 0.03]
# real payments convention: hard decline = won't succeed on bare retry (needs new instrument/manual review)
hard_declines = {"card_expired", "mandate_revoked", "disputed", "fraud_flag"}
payment_methods = ["card", "upi_autopay", "netbanking", "nach"]
segments = ["individual", "smb", "enterprise"]
segment_weights = [0.55, 0.35, 0.10]

actions = ["retry_2h", "retry_24h", "payment_link", "escalation", "stop"]
action_cost = {"retry_2h": 0, "retry_24h": 0, "payment_link": 20, "escalation": 500, "stop": 0}
action_delay_hours = {"retry_2h": 2, "retry_24h": 24, "payment_link": 6, "escalation": 48, "stop": np.nan}
recovery_time_params = {  # (gamma shape, scale) for hours-to-recovery IF recovered, conditional on action
    "retry_2h": (2.0, 3), "retry_24h": (2.0, 14), "payment_link": (2.0, 20),
    "escalation": (2.5, 30), "stop": (1.5, 40),
}

def action_logit(ctx, action):
    base = (
        -1.0
        + ctx["decline_reason"].map({
            "insufficient_funds": 0.3, "card_expired": 0.7, "bank_decline": -0.1,
            "technical_error": 0.9, "mandate_revoked": -0.6, "disputed": -1.2, "fraud_flag": -2.0
        })
        + 2.0 * (ctx["past_payment_success_rate"] - 0.5)
        + 1.6 * (ctx["historical_engagement_score"] - 0.5)
        + 0.5 * (ctx["customer_segment"] == "enterprise")
        - 0.30 * ctx["retry_count_so_far"]
        - 0.02 * ctx["days_overdue"]
        + 0.25 * np.log1p(ctx["previous_recovered_payments"])
        - 0.15 * np.log1p(ctx["previous_failed_payments"])
    )
    if action == "retry_2h":
        eff = np.where(ctx["decline_reason"].isin(["technical_error", "insufficient_funds"]), 0.9, -0.3) - 0.08 * ctx["days_overdue"]
    elif action == "retry_24h":
        eff = np.where(ctx["decline_reason"] == "insufficient_funds", 0.7, -0.1)
    elif action == "payment_link":
        eff = np.where(ctx["decline_reason"].isin(["card_expired", "mandate_revoked"]), 0.9, 0.4)
    elif action == "escalation":
        eff = (0.9 + 0.5 * (ctx["customer_segment"] == "enterprise")
                   + 0.4 * (ctx["decline_reason"] == "disputed")
                   - 1.5 * (ctx["decline_reason"] == "fraud_flag"))
    else:
        eff = -1.8
    return base + eff

def legacy_policy(row):
    if row["decline_reason"] == "fraud_flag": return "stop"
    if row["amount"] > 50000 or row["decline_reason"] == "disputed": return "escalation"
    if row["days_overdue"] <= 1 and row["decline_reason"] in ["technical_error", "insufficient_funds"]: return "retry_2h"
    if row["decline_reason"] == "insufficient_funds": return "retry_24h"
    return "payment_link"

# ---------- customer pool with stable per-customer traits ----------
N_CUSTOMERS = 4000
customers = pd.DataFrame({
    "customer_id": [f"CUST{i:05d}" for i in range(N_CUSTOMERS)],
    "customer_segment": np.random.choice(segments, N_CUSTOMERS, p=segment_weights),
    "customer_tenure_months": np.random.gamma(shape=2.0, scale=8, size=N_CUSTOMERS).round().astype(int),
    "base_success_rate": np.random.beta(a=6, b=2, size=N_CUSTOMERS),
    "base_engagement": np.random.beta(a=2, b=3, size=N_CUSTOMERS),
})

rows = []
now = datetime(2026, 8, 25)
epsilon = 0.20

for _, cust in customers.iterrows():
    n_events = np.random.poisson(1.3) + 1
    ts_offsets = sorted(np.random.uniform(0, 365, n_events), reverse=True)  # days before "now"
    prev_failed, prev_recovered = 0, 0
    for off in ts_offsets:
        payment_timestamp = now - timedelta(days=off)
        decline_reason = np.random.choice(decline_reasons, p=decline_weights)
        row = {
            "customer_id": cust["customer_id"],
            "payment_timestamp": payment_timestamp,
            "amount": round(np.random.lognormal(mean=8.2, sigma=1.1), 2),
            "decline_reason": decline_reason,
            "is_hard_decline": decline_reason in hard_declines,
            "payment_method": np.random.choice(payment_methods),
            "customer_segment": cust["customer_segment"],
            "customer_tenure_months": cust["customer_tenure_months"],
            "days_overdue": int(np.random.exponential(scale=6)),
            "retry_count_so_far": int(np.random.poisson(1.0)),
            "past_payment_success_rate": float(np.clip(cust["base_success_rate"] + np.random.normal(0, 0.03), 0, 1)),
            "historical_engagement_score": float(np.clip(cust["base_engagement"] + np.random.normal(0, 0.03), 0, 1)),
            "previous_failed_payments": prev_failed,
            "previous_recovered_payments": prev_recovered,
        }
        ctx = pd.DataFrame([row])

        # ground-truth counterfactual probabilities for the 4 active actions -- EVAL/AUDIT ONLY, never a model input
        p_retry_2h = 1 / (1 + np.exp(-action_logit(ctx, "retry_2h").iloc[0]))
        p_retry_24h = 1 / (1 + np.exp(-action_logit(ctx, "retry_24h").iloc[0]))
        p_payment_link = 1 / (1 + np.exp(-action_logit(ctx, "payment_link").iloc[0]))
        p_escalation = 1 / (1 + np.exp(-action_logit(ctx, "escalation").iloc[0]))

        chosen = legacy_policy(row) if np.random.rand() > epsilon else np.random.choice(actions)
        p_chosen = {"retry_2h": p_retry_2h, "retry_24h": p_retry_24h,
                    "payment_link": p_payment_link, "escalation": p_escalation,
                    "stop": 1 / (1 + np.exp(-action_logit(ctx, "stop").iloc[0]))}[chosen]

        recovered = int(np.random.binomial(1, p_chosen))
        shape, scale = recovery_time_params[chosen]
        time_to_recovery = round(float(np.random.gamma(shape, scale)), 1) if recovered else np.nan

        row.update({
            "recovery_action": chosen,
            "retry_delay_hours": action_delay_hours[chosen],
            "action_cost": action_cost[chosen],
            "recovery_probability": round(p_chosen, 4),   # TRUE prob of the chosen action -- EVAL ONLY, leaks the label
            "recovered": recovered,
            "recovered_amount": round(row["amount"], 2) if recovered else 0.0,
            "time_to_recovery": time_to_recovery,
            "p_recovery_retry_2h": round(p_retry_2h, 4),
            "p_recovery_retry_24h": round(p_retry_24h, 4),
            "p_recovery_payment_link": round(p_payment_link, 4),
            "p_recovery_escalation": round(p_escalation, 4),
        })
        rows.append(row)

        prev_failed += 1
        if recovered:
            prev_recovered += 1

df = pd.DataFrame(rows)
df.insert(0, "record_id", [f"REC{i:06d}" for i in range(len(df))])
df.to_csv("recovery_full_dataset.csv", index=False)

print("Total records:", len(df))
print("Recovery rate (legacy/logged policy):", round(df['recovered'].mean(), 3))