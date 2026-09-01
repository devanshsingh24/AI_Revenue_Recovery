import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

df = pd.read_csv("recovery_full_dataset.csv")
actions = ["retry_2h", "retry_24h", "payment_link", "escalation", "stop"]
action_cost = {"retry_2h": 0, "retry_24h": 0, "payment_link": 20, "escalation": 500, "stop": 0}

# customer_id / payment_timestamp excluded: raw IDs would let the model memorize
# individual customers rather than generalize; a real system would derive
# recency/frequency features from the timestamp instead of using it raw.
feature_cols = ["amount", "decline_reason", "is_hard_decline", "payment_method", "customer_segment",
                 "customer_tenure_months", "days_overdue", "retry_count_so_far",
                 "past_payment_success_rate", "historical_engagement_score",
                 "previous_failed_payments", "previous_recovered_payments", "recovery_action"]
cat_features = ["decline_reason", "payment_method", "customer_segment", "recovery_action"]

X, y = df[feature_cols], df["recovered"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6,
                            cat_features=cat_features, verbose=False, random_state=42)
model.fit(X_train, y_train)
print("Held-out ROC-AUC:", round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 3))

# score ALL 5 actions for a record, pick the one with the highest net expected value
def ml_policy(row):
    rows = pd.DataFrame([row[feature_cols[:-1]].to_dict() for _ in actions])
    rows["recovery_action"] = actions
    proba = model.predict_proba(rows[feature_cols])[:, 1]
    net_ev = proba * row["amount"] - np.array([action_cost[a] for a in actions])
    return actions[net_ev.argmax()]