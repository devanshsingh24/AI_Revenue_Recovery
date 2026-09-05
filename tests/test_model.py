from pathlib import Path
import unittest

import pandas as pd

from train_policy import ACTIONS, DATA_PATH, FEATURE_COLS, load_model, score_all_actions


class ModelContractTests(unittest.TestCase):
    def test_saved_model_loads_with_its_saved_feature_schema(self):
        model, feature_cols = load_model()

        self.assertEqual(feature_cols, FEATURE_COLS)
        self.assertEqual(model.feature_names_, FEATURE_COLS)

    def test_model_scores_an_existing_dataset_record(self):
        model, _ = load_model()
        row = pd.read_csv(DATA_PATH, nrows=1).iloc[0].to_dict()

        action = score_all_actions(model, row)

        self.assertIn(action, ACTIONS)


if __name__ == "__main__":
    unittest.main()
