import numpy as np
import pandas as pd

np.random.seed(42)
N = 8000

decline_reasons = ["insufficient_funds", "card_expired", "bank_decline",
                    "technical_error", "mandate_revoked", "disputed", "fraud_flag"]
decline_weights = [0.35, 0.15, 0.20, 0.10, 0.10, 0.07, 0.03]
payment_methods = ["card", "upi_autopay", "netbanking", "nach"]
segments = ["individual", "smb", "enterprise"]
segment_weights = [0.55, 0.35, 0.10]

df = pd.DataFrame({
    "record_id": [f"REC{i:05d}" for i in range(N)],
    "amount": np.round(np.random.lognormal(mean=8.2, sigma=1.1, size=N), 2),
    "decline_reason": np.random.choice(decline_reasons, N, p=decline_weights),
    "payment_method": np.random.choice(payment_methods, N),
    "customer_segment": np.random.choice(segments, N, p=segment_weights),
    "customer_tenure_months": np.random.gamma(shape=2.0, scale=8, size=N).round().astype(int),
    "days_overdue": np.random.exponential(scale=6, size=N).round().astype(int).clip(0, 90),
    "past_payment_success_rate": np.random.beta(a=6, b=2, size=N),   # most customers usually pay fine
    "retry_count_so_far": np.random.poisson(lam=1.2, size=N).clip(0, 5),
    "engagement_score": np.random.beta(a=2, b=3, size=N),            # proxy: opened email / clicked link
})

# hidden "true" recovery-probability function — domain logic, not random noise
reason_effect = df["decline_reason"].map({
    "insufficient_funds": 0.3, "card_expired": 0.8, "bank_decline": -0.2,
    "technical_error": 1.0, "mandate_revoked": -0.8, "disputed": -1.4, "fraud_flag": -2.2
})

logit = (
    -0.8
    + reason_effect
    + 2.4 * (df["past_payment_success_rate"] - 0.5)
    + 2.0 * (df["engagement_score"] - 0.5)
    + 0.6 * (df["customer_segment"] == "enterprise")
    + 0.2 * (df["customer_segment"] == "smb")
    - 0.35 * df["retry_count_so_far"]
    - 0.02 * df["days_overdue"]
    - 0.15 * np.log1p(df["amount"]) / 3
    + np.random.normal(0, 0.12, N)   # small residual noise — keeps it non-trivial, not unlearnable
)

prob_recover = 1 / (1 + np.exp(-logit))
df["recovered"] = np.random.binomial(1, prob_recover)   # actual observed outcome, not the same as the probability

df.to_csv("synthetic_recovery_data.csv", index=False)