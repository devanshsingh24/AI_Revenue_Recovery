from __future__ import annotations
from typing import Any, List
from langgraph.graph import END, START, StateGraph
from src.agent.nodes import audit_node, decide_node, diagnose_node, escalate_node, execute_node, policy_validate_node, predict_node
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
