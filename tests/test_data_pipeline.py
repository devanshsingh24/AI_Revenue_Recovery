from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "recovery_full_dataset.csv"


class DatasetContractTests(unittest.TestCase):
    def test_generated_dataset_exists_with_required_training_columns(self):
        self.assertTrue(DATASET.is_file())
        frame = pd.read_csv(DATASET)
        required = {
            "record_id", "customer_id", "amount", "decline_reason",
            "is_hard_decline", "payment_method", "customer_segment",
            "customer_tenure_months", "days_overdue", "retry_count_so_far",
            "past_payment_success_rate", "historical_engagement_score",
            "previous_failed_payments", "previous_recovered_payments",
            "recovery_action", "recovered",
        }
        self.assertTrue(required.issubset(frame.columns))
        self.assertGreater(len(frame), 0)
        self.assertEqual(frame["record_id"].nunique(), len(frame))
        self.assertTrue(frame["recovered"].isin([0, 1]).all())
        self.assertTrue(frame["recovery_action"].isin(
            {"retry_2h", "retry_24h", "payment_link", "escalation", "stop"}
        ).all())


if __name__ == "__main__":
    unittest.main()
