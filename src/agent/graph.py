from __future__ import annotations
from typing import Any, List
from langgraph.graph import END, START, StateGraph
from src.agent.nodes import (
    audit_node, decide_node, diagnose_node, escalate_node,
    execute_node, policy_validate_node, predict_node,
)
from src.agent.state import RecoveryState


def build_graph(model: Any, feature_cols: List[str], razorpay_client: Any):
    graph = StateGraph(RecoveryState)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("predict", lambda state: predict_node(state, model, feature_cols))
    graph.add_node("decide", decide_node)
    graph.add_node("policy_validate", policy_validate_node)
    graph.add_node("execute", lambda state: execute_node(state, razorpay_client))
    graph.add_node("escalate", escalate_node)
    graph.add_node("audit", audit_node)

    graph.add_edge(START, "diagnose")
    graph.add_edge("diagnose", "predict")
    graph.add_edge("predict", "decide")
    graph.add_edge("decide", "policy_validate")

    def route_after_policy(state: RecoveryState) -> str:
        action = state.get("final_action", "stop")
        if action == "escalation":
            return "escalate"
        if action == "stop":
            return "audit"
        return "execute"

    graph.add_conditional_edges("policy_validate", route_after_policy,
        {"execute": "execute", "escalate": "escalate", "audit": "audit"})
    graph.add_edge("execute", "audit")
    graph.add_edge("escalate", "audit")
    graph.add_edge("audit", END)
    return graph.compile()


def build_graph_from_trained_model(razorpay_client: Any):
    """Loads the CatBoost model + feature schema saved by train_policy.py and
    compiles the graph. This is the missing link the audit meant by 'startup
    is blocked by the missing model' — call this once at process startup
    (e.g. FastAPI startup event, or top of run_agent.py) rather than calling
    build_graph() directly, so model-loading isn't duplicated at each call site.
    Raises FileNotFoundError with a clear message if train_policy.py hasn't
    been run yet — this is intentionally NOT swallowed, since starting the
    service without a model is exactly the failure mode being fixed."""
    from train_policy import load_model
    model, feature_cols = load_model()
    return build_graph(model, feature_cols, razorpay_client)


def run_recovery(compiled_graph, state: RecoveryState) -> RecoveryState:
    """Invoke the graph with error containment. Any node exception is caught
    here so a webhook/API caller always gets back a state dict — with an
    audit_record — instead of an unhandled exception with no audit trail.
    Does not catch errors inside execute_node's Razorpay call selectively;
    any exception anywhere in the run is treated the same way, since a
    partial run with no audit record is the failure mode to avoid."""
    try:
        return compiled_graph.invoke(state)
    except Exception as exc:  # noqa: BLE001 - intentionally broad: see docstring
        errors = list(state.get("errors") or [])
        errors.append(f"{type(exc).__name__}: {exc}")
        failed_state: RecoveryState = {
            **state,
            "errors": errors,
            "execution_status": "failed",
            "final_action": state.get("final_action", "stop"),
        }
        return audit_node(failed_state)