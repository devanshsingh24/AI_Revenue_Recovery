# --- Core loop (lines 70–113) ---
import pandas as pd
audit_records = []
override_count = 0
blocked_count = 0
df = pd.read_csv("data/recovery_full_dataset.csv")
for idx, row in df.iterrows():
    # 1. ML recommendation
    ml_result = ml_policy(row, model=model)
    ml_action = ml_result["suggested_action"]

    # 2. Guardrails
    policy_result = apply_guardrails(row, ml_action)
    final_action = policy_result["final_action"]

    if policy_result["status"] == "overridden":
        override_count += 1
    elif policy_result["status"] == "blocked":
        blocked_count += 1

    # 3. Audit record (always written)
    record = create_audit_record(row, ml_result, policy_result)   # ← cleaner
    audit_records.append(record)

    # 4. Execution (optional)
    if execute:
        exec_result = execute_final_action(final_action, row.to_dict())

# Persist audit trail
out_path = save_audit_log(audit_records, "logs/recovery_audit_log.csv")