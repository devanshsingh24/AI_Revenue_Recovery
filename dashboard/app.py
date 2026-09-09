# TODO: add authentication/login before any real client sees this dashboard.
# This app currently has no access control (viewing dashboard only by design).

"""Read-only Streamlit dashboard over the EXISTING AI Revenue Recovery backend.

Data now arrives over HTTP from backend/dashboard_api.py (dashboard/api_client.py)
instead of direct Postgres access. Honesty rule unchanged: `executed` means an
order/payment-link was CREATED, not that money arrived — the greyed
"Confirmed Recovered" placeholder renders for production data. In demo mode the
API serves explicitly-labeled *simulated* recovery figures (see Part C).

Read-only: no write or action buttons exist in this file by design.

Demo mode: `streamlit run dashboard/app.py -- --demo` or `DEMO_MODE=true`.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Ensure both repo root (for `src`, `data`, `train_policy`) and this folder
# (for `api_client`, `batch_eval`) are importable both locally
# (`streamlit run dashboard/app.py` from root) and on Streamlit Cloud
# (script dir on sys.path, root not guaranteed).
_FILE_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _FILE_DIR.parent
for _p in (str(_FILE_DIR), str(_ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import api_client

st.set_page_config(layout="wide", page_title="RR.AI", page_icon="⚡")

_CSS = (Path(__file__).resolve().parent / "styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

CACHE_TTL = 60
REVEAL_DELAY = 0.4  # seconds per audit-trail step in the live reveal

# Restrained chart palette: one neutral accent everywhere, except policy
# statuses which carry their semantic green/amber/red meaning.
CHART_ACCENT = "#2456d6"
POLICY_COLORS = {"allowed": "#16a34a", "overridden": "#d97706", "blocked": "#b42318"}
# Env/flag default: demo on (fresh installs show meaningful data immediately).
_ENV_DEMO = os.getenv("DEMO_MODE", "").lower() == "true" or "--demo" in sys.argv

NAV = {"📊 Overview": "Overview", "🔍 Recovery Detail": "Recovery Detail",
       "🧾 Audit Trail": "Audit Trail", "🔔 Webhook Events": "Webhook Events"}


def _bounds():
    start_d = st.session_state.get("start_date")
    end_d = st.session_state.get("end_date")
    start = datetime.combine(start_d, time.min) if start_d else None
    end = datetime.combine(end_d, time.max) if end_d else None
    return start, end


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


@st.cache_data(ttl=CACHE_TTL)
def cached_overview(mode: str, start_iso: Optional[str], end_iso: Optional[str]) -> dict:
    s = datetime.fromisoformat(start_iso) if start_iso else None
    e = datetime.fromisoformat(end_iso) if end_iso else None
    return api_client.get_kpis(s, e, demo=(mode == "demo"))


@st.cache_data(ttl=CACHE_TTL)
def cached_detail_rows(mode: str, start_iso: Optional[str], end_iso: Optional[str]) -> list:
    s = datetime.fromisoformat(start_iso) if start_iso else None
    e = datetime.fromisoformat(end_iso) if end_iso else None
    return api_client.get_detail_rows(s, e, demo=(mode == "demo"))


@st.cache_data(ttl=CACHE_TTL)
def cached_events(mode: str, limit: int = 100) -> list:
    return api_client.get_events(limit, demo=(mode == "demo"))


@st.cache_data(ttl=CACHE_TTL)
def cached_policy(mode: str, start_iso: Optional[str], end_iso: Optional[str]) -> dict:
    s = datetime.fromisoformat(start_iso) if start_iso else None
    e = datetime.fromisoformat(end_iso) if end_iso else None
    return api_client.get_policy(s, e, demo=(mode == "demo"))


def _pill(value: Optional[str]) -> str:
    v = str(value or "unknown")
    tone = "pill-grey"
    if v in ("allowed", "executed", "processed"):
        tone = "pill-green"
    elif v in ("overridden", "escalated", "escalation", "processed_with_errors"):
        tone = "pill-amber"
    elif v in ("blocked", "failed", "stop", "not_executed"):
        tone = "pill-red"
    return f'<span class="pill {tone}">{v}</span>'


def _kpi_card(label: str, value: str, icon: str, tone: str, caption: str = "",
              placeholder: bool = False, help_text: str = "") -> str:
    cls = "kpi-card placeholder" if placeholder else "kpi-card"
    title = f'title="{help_text}"' if help_text else ""
    return (f'<div class="{cls}" {title}><div class="kpi-label">{label}</div>'
            f'<div class="kpi-row"><div class="kpi-value">{value}</div>'
            f'<div class="kpi-icon {tone}">{icon}</div></div>'
            f'<div class="kpi-cap">{caption}</div></div>')


def _delta_caption(cur: float, prev: Optional[float]) -> str:
    if prev is None or prev == 0:
        return "no prior-period baseline"
    pct = (cur - prev) / abs(prev) * 100.0
    return f"{pct:+.1f}% vs prior period"


def _prior_delta(metric: str, start: Optional[datetime], end: Optional[datetime],
                 cur: float) -> str:
    """Honest delta vs the previous equal-length period (real recomputation)."""
    if not start or not end or end <= start:
        return "set a date range for trend"
    span = (end - start) + timedelta(days=1)
    try:
        prev = cached_overview(MODE, _iso(start - span), _iso(start - timedelta(days=1)))
        return _delta_caption(cur, float(prev.get(metric, 0) or 0))
    except Exception:
        return "prior period unavailable"


def _require_api(demo: bool):
    try:
        api_client.get_kpis(None, None, demo=demo)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()


if "sidebar_hidden" not in st.session_state:
    st.session_state["sidebar_hidden"] = False
if st.sidebar.button("<<", key="sidebar_hide",
                     help="Hide the sidebar so the main content takes the full width"):
    st.session_state["sidebar_hidden"] = True
    st.rerun()
st.sidebar.markdown('<div class="avatar">RR</div>', unsafe_allow_html=True)
st.sidebar.markdown("### RR.AI")
st.sidebar.caption("Revenue Recovery — read-only operations dashboard")
st.sidebar.caption("Payment events via Razorpay")
demo_on = st.sidebar.toggle("Demo mode", value=True,
                            help="On: curated sample data via demo=true. Off: live production data.")
DEMO_MODE = bool(demo_on or _ENV_DEMO)
MODE = "demo" if DEMO_MODE else "live"
_require_api(DEMO_MODE)
if DEMO_MODE:
    st.sidebar.info("Demo mode — curated sample data, not real customers.")
_order = list(NAV.keys())
if "page" not in st.session_state:
    st.session_state["page"] = _order[0]
choice = st.sidebar.radio("Navigate", _order,
                          index=_order.index(st.session_state["page"]))
if choice != st.session_state["page"]:
    st.session_state["page"] = choice
page = NAV[choice]
st.sidebar.header("Date range")
st.sidebar.date_input("Start date (YYYY/MM/DD)", value=None, key="start_date", format="YYYY/MM/DD")
st.sidebar.date_input("End date (YYYY/MM/DD)", value=None, key="end_date", format="YYYY/MM/DD")
st.sidebar.caption("Pick a date — format YYYY/MM/DD")
if st.sidebar.button("Refresh now"):
    st.cache_data.clear()

if st.session_state.get("sidebar_hidden"):
    st.markdown("<style>[data-testid='stSidebar']{display:none !important;}</style>",
                unsafe_allow_html=True)
    if st.button(">>", key="sidebar_show",
                 help="Bring back the sidebar navigation"):
        st.session_state["sidebar_hidden"] = False
        st.rerun()

if DEMO_MODE:
    st.info("Demo mode is on: curated sample data with realistic-looking (fake) records. No real customer data shown.")

_top_l, _top_r = st.columns([8, 2])
with _top_l:
    st.caption("Step between pages with Previous / Next.")
with _top_r:
    _prev_col, _next_col = st.columns(2)
    with _prev_col:
        if st.button("← Previous", key="page_prev", help="Previous page"):
            _cur = _order.index(st.session_state.get("page", _order[0]))
            st.session_state["page"] = _order[(_cur - 1) % len(_order)]
            st.rerun()
    with _next_col:
        if st.button("Next →", key="page_next", help="Next page"):
            _cur = _order.index(st.session_state.get("page", _order[0]))
            st.session_state["page"] = _order[(_cur + 1) % len(_order)]
            st.rerun()

if page == "Overview":
    st.subheader("Overview")
    try:
        start, end = _bounds()
        data = cached_overview(MODE, _iso(start), _iso(end))
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    total = data["total_failed"]
    final = data["final_action_counts"]
    policy = data["policy_status_counts"]
    if total == 0:
        st.info("No failed payments in this range yet (fresh install or filter too narrow).")
    actions_taken = data["actions_taken"]
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(_kpi_card("Failed payments", f"{total}", "📉", "blue",
                          _prior_delta("total_failed", start, end, total)), unsafe_allow_html=True)
    k2.markdown(_kpi_card("Recovery Actions Taken", f"{actions_taken}", "⚡", "green",
                          _prior_delta("actions_taken", start, end, actions_taken),
                          help_text="Retries + payment links created. NOT confirmed money received."),
                unsafe_allow_html=True)
    k3.markdown(_kpi_card("Action Success Rate", f"{data['action_rate_pct']:.1f}%", "🎯", "green",
                          "% not stopped",
                          help_text="% of failures where the system took retry/payment_link rather than stop/escalate."),
                unsafe_allow_html=True)
    k4.markdown(_kpi_card("Escalated %", f"{data['escalated_pct']:.1f}%", "🧑‍💼", "amber",
                          "needs human review"), unsafe_allow_html=True)
    k5.markdown(_kpi_card("Stopped (blocked) %", f"{data['stopped_pct']:.1f}%", "🛑", "red",
                          "blocked by policy"), unsafe_allow_html=True)
    if "simulated_recovery_rate" in data:
        k6.markdown(_kpi_card("Recovery Rate", f"{data['simulated_recovery_rate']:.1f}%", "💹", "green",
                              "(simulated demo outcome) · "
                              f"₹{data['simulated_recovered_amount']:,.0f} across "
                              f"{data['simulated_recovered_count']} payments",
                              help_text="SIMULATED demo outcome (coin-flip weighted by model probability). "
                                        "Not a measured result."),
                    unsafe_allow_html=True)
    else:
        k6.markdown(_kpi_card("Confirmed Recovered ₹", "—", "⏳", "grey",
                              "coming soon — pending capture confirmation",
                              placeholder=True,
                              help_text="Pending capture-confirmation integration (not yet available). "
                                        "Executed actions are unconfirmed attempts, not receipts."),
                    unsafe_allow_html=True)
    st.divider()
    b1, b2 = st.columns(2)
    with b1:
        st.markdown('<div class="card"><b>Live Batch Evaluation</b> — demo mode only<br/>'
                    '<span style="color:#667085;font-size:12px">Fresh synthetic records through the '
                    'real model + policy, appended to the demo database.</span></div>',
                    unsafe_allow_html=True)
        if DEMO_MODE:
            n = st.selectbox("Records per run", [50, 100, 500], index=1, key="batch_n")
            if st.button("Run Live Batch Evaluation", key="batch_run"):
                import time as _time

                try:
                    from dashboard import batch_eval
                except ImportError:
                    import batch_eval

                try:
                    bar = st.progress(0, text="Scoring records...")
                    summary = None
                    for done, total, summ in batch_eval.run_batch(int(n)):
                        bar.progress(done / total, text=f"Processing {done}/{total}...")
                        if summ is not None:
                            summary = summ
                    bar.empty()
                    st.cache_data.clear()
                except Exception as exc:  # noqa: BLE001 - batch must never take down the page
                    bar.empty()
                    st.error(f"Batch run failed ({type(exc).__name__}: {exc}). "
                             f"Earlier chunks may already be committed — check the demo database and retry.")
                if summary:
                    st.success(f"Processed {summary['processed']} records in "
                               f"{summary['elapsed_s']}s — simulated recovered "
                               f"{summary['simulated_recovered']}. KPIs refresh on next load.")
                    st.write({"action_breakdown": summary["final_action_counts"],
                              "policy_breakdown": summary["policy_status_counts"]})
        else:
            st.caption("Batch evaluation is disabled in production mode.")
    with b2:
        st.markdown('<div class="card"><b>Policy Configuration</b> — live values from '
                    '`src/guardrails.py`<br/><span style="color:#667085;font-size:12px">'
                    'Read at runtime; edits to guardrails.py appear here with no dashboard change.'
                    '</span></div>', unsafe_allow_html=True)
        try:
            from src.guardrails import HIGH_VALUE_THRESHOLD, RETRY_LIMIT

            st.write(f"Max retries before forced payment_link: {RETRY_LIMIT}")
            st.write(f"High-value escalation threshold: ₹{HIGH_VALUE_THRESHOLD:,}")
            st.caption("fraud → always stop · disputed → always escalate · "
                       "mandate_revoked/card_expired → forced payment_link")
        except Exception as exc:
            st.error(f"Could not read policy thresholds. ({type(exc).__name__})")
    try:
        _model_info = api_client.get_model_info()
    except RuntimeError:
        _model_info = {"available": False}
    if _model_info.get("available"):
        st.markdown('<div class="card">Model held-out ROC-AUC: '
                    f"{_model_info['held_out_roc_auc']} "
                    f"(trained {_model_info.get('trained_at', 'unknown date')})</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="card">Model metrics not available — retrain to populate</div>',
                    unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><b>Failed payments over time</b></div>', unsafe_allow_html=True)
        try:
            rows = cached_detail_rows(MODE, _iso(start), _iso(end))
        except RuntimeError as exc:
            st.error(str(exc))
            rows = []
        if rows:
            df = pd.DataFrame(rows)
            df["day"] = pd.to_datetime(df["created_at"]).dt.date
            st.area_chart(df.groupby("day").size(), color=CHART_ACCENT)
        else:
            st.info("No records in range for trend.")
        st.markdown('<div class="card"><b>Policy status (allowed vs overridden vs blocked)</b></div>',
                    unsafe_allow_html=True)
        if policy:
            import altair as alt

            _pol = pd.DataFrame({"status": list(policy.keys()), "count": list(policy.values())})
            st.altair_chart(alt.Chart(_pol).mark_bar().encode(
                x=alt.X("status:N", title=None),
                y=alt.Y("count:Q", title=None),
                color=alt.Color("status:N",
                                scale=alt.Scale(domain=list(POLICY_COLORS.keys()),
                                                range=list(POLICY_COLORS.values())),
                                legend=None),
            ), width="stretch")
        else:
            st.info("No policy decisions in range.")
    with c2:
        st.markdown('<div class="card"><b>Count by decline_reason</b></div>', unsafe_allow_html=True)
        if data["decline_reason_counts"]:
            st.bar_chart(pd.Series(data["decline_reason_counts"]), color=CHART_ACCENT)
        else:
            st.info("No payments in range.")
        st.markdown('<div class="card"><b>Count by final_action</b></div>', unsafe_allow_html=True)
        if final:
            st.bar_chart(pd.Series(final), color=CHART_ACCENT)
        else:
            st.info("No recovery attempts in range.")
        st.markdown('<div class="card"><b>Amount attempted by final_action (₹ at risk, NOT recovered)</b></div>',
                    unsafe_allow_html=True)
        if data["amount_attempted_by_final_action"]:
            st.bar_chart(pd.Series(data["amount_attempted_by_final_action"]), color=CHART_ACCENT)
        else:
            st.info("No amounts in range.")
    if "model_open" not in st.session_state:
        st.session_state["model_open"] = False
    _open = st.session_state["model_open"]
    if st.button(("Hide ▲ " if _open else "Show ▼ ") + "what the model considered",
                 key="model_toggle",
                 help="Expand or collapse the model detail section"):
        st.session_state["model_open"] = not _open
        st.rerun()
    st.markdown("**What did the model actually consider?**")
    if st.session_state["model_open"]:
        try:
            conf = cached_policy(MODE, _iso(start), _iso(end))["model_stats"]
        except RuntimeError as exc:
            st.error(str(exc))
            conf = None
        if conf and conf["records_examined"]:
            st.write(f"Records examined: {conf['records_examined']}")
            if conf["avg_ml_confidence"] is not None:
                st.write(f"Average ml_confidence: {conf['avg_ml_confidence']:.3f}")
            if conf["mean_action_probabilities"]:
                st.bar_chart(pd.Series(conf["mean_action_probabilities"]), color=CHART_ACCENT)
        else:
            st.info("No scored audit records in range.")

elif page == "Recovery Detail":
    st.subheader("Recovery Detail")
    try:
        start, end = _bounds()
        rows = cached_detail_rows(MODE, _iso(start), _iso(end))
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    if not rows:
        st.info("No records in range.")
    else:
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            width="stretch",
            column_config={
                "payment_id": st.column_config.TextColumn("Payment", width="medium"),
                "customer_id": st.column_config.TextColumn("Customer", width="small"),
                "amount": st.column_config.NumberColumn("Amount (₹)", format="₹%.0f"),
                "decline_reason": st.column_config.TextColumn("Reason", width="medium"),
                "ml_suggested_action": st.column_config.TextColumn("ML suggestion", width="small"),
                "final_action": st.column_config.TextColumn("Final action", width="small"),
                "policy_status": st.column_config.TextColumn("Policy", width="small"),
                "execution_status": st.column_config.TextColumn("Execution", width="small"),
            },
        )
        st.caption("Status key: "
                   + _pill("allowed") + " " + _pill("overridden") + " " + _pill("blocked") + " | "
                   + _pill("executed") + " " + _pill("escalated") + " " + _pill("failed"))
        ids = [r["payment_id"] for r in rows if r.get("payment_id")]
        selected = st.selectbox("Drill down by payment_id", ids)
        if selected:
            try:
                payload = api_client.get_payment(selected, demo=DEMO_MODE)
                detail = payload.get("detail") or {}
            except RuntimeError as exc:
                st.error(str(exc))
                detail = {}
            if detail:
                st.markdown(f"### {selected}  {_pill(detail.get('policy_status'))} "
                            f"{_pill(detail.get('execution_status'))}", unsafe_allow_html=True)
                st.write(f"Amount: ₹{detail.get('amount')} {detail.get('currency')} · "
                         f"Reason: {detail.get('decline_reason')}")
                st.write(f"ML suggested: {detail.get('ml_suggested_action')} → "
                         f"Final: {detail.get('final_action')}")
                st.markdown("**Diagnosis**")
                diag = detail.get("diagnosis") or {}
                st.write(diag.get("classification", "—"))
                st.markdown("**Action probabilities (what the model considered)**")
                probs = detail.get("action_probabilities") or {}
                if probs:
                    st.bar_chart(pd.Series({k: float(v) for k, v in probs.items()}), color=CHART_ACCENT)
                st.markdown("**Expected net recovery per action (model estimate, not receipts)**")
                ev = detail.get("expected_net_recovery") or {}
                if ev:
                    st.bar_chart(pd.Series({k: float(v) for k, v in ev.items()}), color=CHART_ACCENT)
                st.markdown("**Policy reason**")
                st.write(detail.get("policy_reason") or "—")
                st.markdown("**Execution / escalation result**")
                st.write({k: v for k, v in
                          {**detail.get("execution_result", {}),
                           **detail.get("escalation_result", {})}.items()})

elif page == "Audit Trail":
    st.subheader("Audit Trail")
    q_payment = st.text_input("Search payment_id", key="audit_payment",
                                placeholder="e.g. pay_seed_000001")
    q_customer = st.text_input("Search customer_id", key="audit_customer",
                               placeholder="e.g. CUST02787")
    if st.button("Show trail"):
        if not q_payment and not q_customer:
            st.info("Enter a payment_id or customer_id.")
        else:
            try:
                if q_payment:
                    trail = api_client.get_trail(payment_id=q_payment, demo=DEMO_MODE)
                else:
                    trail = api_client.get_trail(customer_id=q_customer, demo=DEMO_MODE)
            except RuntimeError as exc:
                st.error(str(exc))
                trail = []
            if not trail:
                st.info("No trail found for that entity. Check the ID matches the active "
                        "database (demo IDs look like pay_seed_000001; flip the Demo mode "
                        "toggle if you are searching the other dataset).")
            for entry in trail:
                det = entry.get("detail") or {}
                st.markdown(f"### Payment {entry['payment_id']}  "
                            f"{_pill(det.get('policy_status'))} {_pill(det.get('execution_status'))}",
                            unsafe_allow_html=True)
                import html as _html
                import time as _time

                slot = st.empty()
                shown: list = []
                for step in entry["steps"]:
                    shown.append(step)
                    slot.markdown(
                        '<div class="term">' + "".join(
                            f"<div><span class=\"term-prompt\">❯</span> "
                            f"<span class=\"term-stage\">{_html.escape(str(s['stage']))}</span> "
                            f"<span class=\"term-text\">{_html.escape(str(s['summary']))}</span></div>"
                            for s in shown) + "</div>",
                        unsafe_allow_html=True)
                    _time.sleep(REVEAL_DELAY)

else:
    st.subheader("Webhook Events")
    try:
        events = cached_events(MODE)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    if not events:
        st.info("No webhook events stored yet.")
    else:
        df = pd.DataFrame(events)
        if "received_at" in df.columns:
            df["received_at"] = pd.to_datetime(df["received_at"])
        st.dataframe(
            df,
            width="stretch",
            column_config={
                "event_id": st.column_config.TextColumn("Event", width="medium"),
                "received_at": st.column_config.DatetimeColumn("Received at"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "payment_id": st.column_config.TextColumn("Payment", width="medium"),
            },
        )
        st.caption("Status values: processed / processed_with_errors / ignored / duplicate_ignored. "
                   "Evidence the system is receiving and handling events.")

st.divider()
st.markdown(
    '<div class="footer">RR.AI · <a href="https://github.com/devanshsingh24" target="_blank">'
    '<svg height="18" width="18" viewBox="0 0 24 24" fill="#101828" aria-hidden="true"><path d="M12 .297c-6.63 '
    '0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 '
    '18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 '
    '3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 '
    '0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 '
    '2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 '
    '3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>'
    'github.com/devanshsingh24</a></div>',
    unsafe_allow_html=True,
)
