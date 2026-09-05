import unittest

from src.agent.graph import build_graph
from train_policy import FEATURE_COLS, load_model


class FakeRazorpayClient:
    def __init__(self):
        self.calls = []

    def execute_recovery_action(self, **kwargs):
        self.calls.append(kwargs)
        return {"mode": "dry_run", "operation": kwargs["action"]}


def valid_state(**overrides):
    state = {
        "event_id": "evt-test", "event_name": "payment.failed",
        "payment_id": "pay-test", "record_id": "record-test",
        "amount": 1_800.0, "currency": "INR",
        "decline_reason": "insufficient_funds", "payment_method": "card",
        "customer_segment": "individual", "customer_tenure_months": 12,
        "days_overdue": 0, "retry_count_so_far": 0,
        "past_payment_success_rate": 0.8, "historical_engagement_score": 0.6,
        "previous_failed_payments": 1, "previous_recovered_payments": 2,
    }
    state.update(overrides)
    return state


class AgentGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _ = load_model()

    def graph_with_fake_client(self):
        client = FakeRazorpayClient()
        return build_graph(self.model, FEATURE_COLS, client), client

    def test_graph_scores_actions_and_records_execution(self):
        graph, client = self.graph_with_fake_client()
        result = graph.invoke(valid_state())

        self.assertEqual(set(result["action_probabilities"]), {
            "retry_2h", "retry_24h", "payment_link", "escalation", "stop"
        })
        self.assertIn(result["final_action"], result["action_probabilities"])
        self.assertIn("audit_record", result)
        if result["final_action"] in {"retry_2h", "retry_24h", "payment_link"}:
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(result["execution_status"], "executed")

    def test_fraud_is_stopped_without_external_execution(self):
        graph, client = self.graph_with_fake_client()
        result = graph.invoke(valid_state(decline_reason="fraud_flag"))

        self.assertEqual(result["final_action"], "stop")
        self.assertEqual(result["policy_status"], "blocked")
        self.assertEqual(result["execution_status"], "not_executed")
        self.assertEqual(client.calls, [])

    def test_dispute_is_escalated_without_external_execution(self):
        graph, client = self.graph_with_fake_client()
        result = graph.invoke(valid_state(decline_reason="disputed"))

        self.assertEqual(result["final_action"], "escalation")
        self.assertEqual(result["execution_status"], "escalated")
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
