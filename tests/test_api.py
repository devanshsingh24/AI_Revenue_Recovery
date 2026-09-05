import unittest

from fastapi.testclient import TestClient

import backend.main as main_module


class FakeHandler:
    def __init__(self, result=None, error=None):
        self.result = result or {"status": "processed"}
        self.error = error
        self.calls = []

    def handle(self, raw_body, headers):
        self.calls.append((raw_body, headers))
        if self.error is not None:
            raise self.error
        return self.result


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.orig = main_module.webhook_handler

    def tearDown(self):
        main_module.webhook_handler = self.orig

    def client(self):
        return TestClient(main_module.app)

    def test_health_returns_ok(self):
        out = self.client().get("/health")
        self.assertEqual(out.status_code, 200)
        self.assertEqual(out.json()["status"], "ok")
        self.assertIn(out.json()["razorpay_mode"], {"dry_run", "test_mode"})

    def test_webhook_503_when_not_initialized(self):
        main_module.webhook_handler = None
        out = self.client().post(
            "/webhooks/razorpay", content=b"{}",
            headers={"x-razorpay-signature": "sig"},
        )
        self.assertEqual(out.status_code, 503)

    def test_webhook_200_delegates_to_handler(self):
        fake = FakeHandler(result={"status": "processed", "payment_id": "pay-1"})
        main_module.webhook_handler = fake
        out = self.client().post(
            "/webhooks/razorpay", content=b'{"event":"payment.failed"}',
            headers={"x-razorpay-signature": "sig", "x-razorpay-event-id": "evt-1"},
        )
        self.assertEqual(out.status_code, 200)
        self.assertEqual(out.json()["status"], "processed")
        self.assertEqual(len(fake.calls), 1)
        raw, headers = fake.calls[0]
        self.assertEqual(raw, b'{"event":"payment.failed"}')
        self.assertEqual(headers["x-razorpay-signature"], "sig")

    def test_webhook_400_on_value_error(self):
        main_module.webhook_handler = FakeHandler(error=ValueError("Invalid Razorpay webhook signature."))
        out = self.client().post(
            "/webhooks/razorpay", content=b"{}",
            headers={"x-razorpay-signature": "bad"},
        )
        self.assertEqual(out.status_code, 400)

    def test_webhook_500_on_unexpected_error(self):
        main_module.webhook_handler = FakeHandler(error=RuntimeError("boom"))
        out = self.client().post(
            "/webhooks/razorpay", content=b"{}",
            headers={"x-razorpay-signature": "sig"},
        )
        self.assertEqual(out.status_code, 500)


if __name__ == "__main__":
    unittest.main()
