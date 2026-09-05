import importlib
import os
import sys
import unittest


def load_config(**values):
    """Import config with an intentionally controlled environment.

    The dev `.env` file is ignored here so tests simulate exactly the
    variables passed in — otherwise a real local `.env` would refill
    popped keys on reimport and missing-secret cases could never raise.
    """
    from unittest import mock

    for key in (
        "RAZORPAY_KEY_ID",
        "RAZORPAY_KEY_SECRET",
        "RAZORPAY_WEBHOOK_SECRET",
        "RAZORPAY_DRY_RUN",
        "DATABASE_URL",
    ):
        os.environ.pop(key, None)
    for key, value in values.items():
        os.environ[key] = value
    sys.modules.pop("src.config", None)
    with mock.patch("dotenv.load_dotenv", lambda *args, **kwargs: False):
        return importlib.import_module("src.config")


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.environment = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.environment)
        sys.modules.pop("src.config", None)

    def test_dry_run_requires_a_webhook_secret(self):
        with self.assertRaisesRegex(RuntimeError, "RAZORPAY_WEBHOOK_SECRET"):
            load_config(RAZORPAY_DRY_RUN="true")

    def test_live_mode_requires_razorpay_api_credentials(self):
        with self.assertRaisesRegex(RuntimeError, "RAZORPAY_KEY_ID"):
            load_config(
                RAZORPAY_DRY_RUN="false",
                RAZORPAY_WEBHOOK_SECRET="test-webhook-secret",
            )

    def test_dry_run_allows_missing_api_credentials(self):
        config = load_config(
            RAZORPAY_DRY_RUN="true",
            RAZORPAY_WEBHOOK_SECRET="test-webhook-secret",
            DATABASE_URL="postgresql+psycopg2://example",
        )

        self.assertTrue(config.settings.dry_run)
        self.assertIsNone(config.settings.razorpay_key_id)
        self.assertEqual(config.settings.database_url, "postgresql+psycopg2://example")
