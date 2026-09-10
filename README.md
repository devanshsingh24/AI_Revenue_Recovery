# AI Revenue Recovery System

> **Live demo:** dashboard is deployed at
> https://airevenuerecovery-ebgxx8nmhtm29jwjahk4ll.streamlit.app/
>
> Backend (FastAPI on Render) URL is not hardcoded — the dashboard reads it
> from the `API_BASE_URL` secret/env var.

## Try the Demo on Your Machine (localhost)

No Razorpay account or credentials needed — the demo runs fully offline in
dry-run mode with a pre-trained model and disposable SQLite databases.

**Prerequisites:** Python 3.11 and Git.

```bash
git clone https://github.com/devanshsingh24/AI_Revenue_Recovery.git
cd AI_Revenue_Recovery
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate
pip install -r requirements.txt
```

**1. Create your local env file** (never commit it — `.env` is git-ignored):

```bash
cp .env.example .env
```

Then open `.env` and make sure these two lines are set (any random string
works as the webhook secret for the demo — it is only checked, never used
to reach Razorpay):

```
RAZORPAY_DRY_RUN=true
RAZORPAY_WEBHOOK_SECRET=local-demo-dummy-secret
```

**2. Create the databases** (one throwaway command; safe to re-run):

```bash
# Windows (PowerShell):
$env:RAZORPAY_WEBHOOK_SECRET = "local-demo-dummy-secret"
$env:RAZORPAY_DRY_RUN = "true"
$env:DATABASE_URL = "sqlite:///./local.db"
$env:DEMO_DATABASE_URL = "sqlite:///./demo_revenue_recovery.db"
python -c "from src.db import init_db; init_db()"
python scripts/seed_demo_data.py --n 1200
```

```bash
# macOS / Linux:
export RAZORPAY_WEBHOOK_SECRET="local-demo-dummy-secret" RAZORPAY_DRY_RUN="true"
export DATABASE_URL="sqlite:///./local.db" DEMO_DATABASE_URL="sqlite:///./demo_revenue_recovery.db"
python -c "from src.db import init_db; init_db()"
python scripts/seed_demo_data.py --n 1200
```

**3. Start the backend API** (terminal 1 — keep it running):

```bash
uvicorn backend.dashboard_api:dashboard_app --port 8001
```

Check it works: open http://127.0.0.1:8001/health (expect `{"status":"ok"}`).

**4. Start the dashboard** (terminal 2 — keep it running):

```bash
streamlit run dashboard/app.py
```

**5. View the demo:** open http://localhost:8501 in your browser.
Keep **Demo mode ON** in the sidebar (it is on by default) — you will see
Overview KPIs, Recovery Detail, Audit Trail, and Webhook Events from the
seeded demo data. Flip Demo mode OFF only if you have wired a real
production database; otherwise leave it on.

**Stopping:** press `Ctrl+C` in both terminals. To start over with fresh
demo data, just re-run the seed command from step 2 (it rebuilds the demo
database from scratch — production data is never touched).

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