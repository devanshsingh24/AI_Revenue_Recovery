import unittest

from src.guardrails import apply_guardrails, evaluate_policy


class GuardrailTests(unittest.TestCase):
    def test_fraud_is_blocked(self):
        result = evaluate_policy(decline_reason="fraud_flag", amount=100, retry_count_so_far=0, ml_action="retry_2h")
        self.assertEqual((result["final_action"], result["status"]), ("stop", "blocked"))

    def test_dispute_is_escalated(self):
        result = evaluate_policy(decline_reason="disputed", amount=100, retry_count_so_far=0, ml_action="stop")
        self.assertEqual((result["final_action"], result["status"]), ("escalation", "overridden"))

    def test_retry_limit_uses_payment_link(self):
        result = evaluate_policy(decline_reason="insufficient_funds", amount=100, retry_count_so_far=3, ml_action="retry_24h")
        self.assertEqual(result["final_action"], "payment_link")

    def test_high_value_requires_escalation(self):
        result = evaluate_policy(decline_reason="bank_decline", amount=50_001, retry_count_so_far=0, ml_action="payment_link")
        self.assertEqual(result["final_action"], "escalation")

    def test_mandate_and_card_rules_block_retries(self):
        for reason in ("mandate_revoked", "card_expired"):
            with self.subTest(reason=reason):
                result = evaluate_policy(decline_reason=reason, amount=100, retry_count_so_far=0, ml_action="retry_2h")
                self.assertEqual(result["final_action"], "payment_link")

    def test_allowed_recommendation_is_preserved_and_row_adapter_matches(self):
        direct = evaluate_policy(decline_reason="technical_error", amount=100, retry_count_so_far=0, ml_action="retry_2h")
        adapted = apply_guardrails({"decline_reason": "technical_error", "amount": 100, "retry_count_so_far": 0}, "retry_2h")
        self.assertEqual(direct, adapted)
        self.assertEqual(direct["status"], "allowed")


if __name__ == "__main__":
    unittest.main()
