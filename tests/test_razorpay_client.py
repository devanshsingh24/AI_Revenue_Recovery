import hashlib
import hmac
import importlib
import os
import sys
import unittest


def client_module():
    os.environ["RAZORPAY_DRY_RUN"] = "true"
    os.environ["RAZORPAY_WEBHOOK_SECRET"] = "test-webhook-secret"
    os.environ.pop("RAZORPAY_KEY_ID", None)
    os.environ.pop("RAZORPAY_KEY_SECRET", None)
    sys.modules.pop("src.razorpay_client", None)
    sys.modules.pop("src.config", None)
    return importlib.import_module("src.razorpay_client")


class RazorpayClientTests(unittest.TestCase):
    def setUp(self):
        self.environment = dict(os.environ)
        self.client = client_module().RazorpayClient()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.environment)
        sys.modules.pop("src.razorpay_client", None)
        sys.modules.pop("src.config", None)

    def test_signature_verification_uses_raw_body(self):
        body = b'{"event":"payment.failed"}'
        signature = hmac.new(b"test-webhook-secret", body, hashlib.sha256).hexdigest()
        self.assertTrue(self.client.verify_webhook_signature(body, signature))
        self.assertFalse(self.client.verify_webhook_signature(body, "invalid"))

    def test_dry_run_actions_do_not_call_razorpay(self):
        retry = self.client.execute_recovery_action("retry_2h", 1_000, "INR", "record-1", "test")
        link = self.client.execute_recovery_action("payment_link", 1_000, "INR", "record-1", "test")

        self.assertEqual(retry["mode"], "dry_run")
        self.assertEqual(retry["scheduled_for_hours"], 2)
        self.assertEqual(link["operation"], "create_payment_link")

    def test_non_executable_action_is_rejected(self):
        with self.assertRaises(ValueError):
            self.client.execute_recovery_action("stop", 1_000, "INR", "record-1", "test")


if __name__ == "__main__":
    unittest.main()
