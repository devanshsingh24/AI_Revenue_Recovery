# AI Revenue Recovery System

## Architecture Overview

```
Webhook → Diagnose → Predict (ML) → Decide → Policy → Execute → Audit
```

## 6 Core Layers

### Layer 1: Data + ML Model
**Purpose:** Predict recovery probability for each action

- **Input:** Payment context (amount, reason, method, customer history, etc.)
- **Model:** CatBoost trained on synthetic recovery data
- **Output:** For each action, predict probability of recovery
  - `retry_2h` → 0.41
  - `retry_24h` → 0.73
  - `payment_link` → 0.54
  - `escalation` → 0.30
  - `stop` → 0.00

**Calculate:** Expected Value = P(recovery) × amount - action_cost

**Key Rule:** ML layer is dumb about Razorpay. It only outputs:
- `ml_suggested_action`
- `action_probabilities`
- `expected_net_recovery`
- `ml_confidence`

**File Structure:**
```
ml/
├── generate_data.py      # Create synthetic dataset
├── train.py              # Train & save model
├── evaluate.py           # Compare policies
└── inference.py          # Reusable scoring
```

---

### Layer 2: Policy Guardrails
**Purpose:** Validate if ML suggestion is allowed

**Checks:**
- Is this fraud?
- Is this disputed?
- Retry limit exceeded?
- Amount too high?
- Payment type compatible?

**Golden Rule:**
```
ML Suggestion → Policy Validator → final_action
```

**Output:**
- `final_action` (approved or modified)
- `policy_status` (ALLOWED / BLOCKED)
- `policy_reason` (why)

**Recommended File:**
```
src/policy/guardrails.py
```

---

### Layer 3: LangGraph Agent
**Purpose:** Orchestrate the workflow

The agent is NOT your intelligence—it's your workflow manager.

**Node Flow:**
1. **diagnose_node** → What kind of failure? (e.g., insufficient_funds)
2. **predict_node** → ML scores each action
3. **decide_node** → Record reasoning (lightweight, doesn't authorize)
4. **policy_validate_node** → Hard gate: approve or block
5. **execute_node** → Execute approved action only
6. **escalate_node** → Handle escalations
7. **audit_node** → Log everything for explainability

**Each node updates `RecoveryState`** (single source of truth):
```
Webhook → RecoveryState → Node A modifies → Node B modifies → ... → Audit
```

---

### Layer 4: Razorpay Adapter
**Purpose:** Isolate API calls from AI logic

```
Agent says: "retry_24h"
        ↓
Adapter translates to Razorpay action
        ↓
Razorpay API executes
```

**File:** `razorpay_client.py`

**Important:** `retry_24h` means "schedule next attempt", not "charge customer".

---

### Layer 5: Webhook Handler
**Purpose:** Convert external events to internal state

```
Razorpay Event → normalize → extract payment_id & amount → RecoveryState → Agent
```

**File:** `webhook_handler.py`

**Future Improvement:** Fetch customer history from your database instead of Razorpay notes:
```
customer_id → PostgreSQL → payment history → RecoveryState
```

---

### Layer 6: Database & Audit
**Purpose:** Store history & prove decisions

**Two Uses:**
1. **Historical data** → Build ML features (past success rate, previous failures)
2. **Audit trail** → Explain what agent did

**Minimal Schema:**
- `customers` (tenure, engagement)
- `payments` (amount, method, reason)
- `recovery_attempts` (action taken, result)
- `audit_logs` (full decision record)

**Audit Log Fields:**
```
event_id, payment_id, amount, decline_reason,
ml_suggested_action, ml_confidence, action_probabilities,
expected_net_recovery, policy_status, policy_reason,
final_action, execution_status, execution_result, timestamp
```

---

## RecoveryState: The Bridge

`RecoveryState` is your shared data structure connecting all layers:

```python
class RecoveryState:
    # Input from webhook
    payment_id
    amount
    decline_reason
    
    # Features from database
    customer_tenure_months
    past_payment_success_rate
    previous_failed_payments
    
    # ML output
    ml_suggested_action
    action_probabilities
    expected_net_recovery
    
    # Policy output
    final_action
    policy_status
    policy_reason
    
    # Execution result
    execution_status
    timestamp
```

---

## Key Principles

✅ **ML proposes** → **Policy approves** → **Agent executes**

✅ **Separate concerns:** ML ≠ Policy ≠ Razorpay API

✅ **Single source of truth:** All nodes read/write `RecoveryState`

✅ **Never skip policy:** ML can never directly set `final_action`

✅ **Explainable:** Audit log shows what each layer decided and why