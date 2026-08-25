import unittest
from fastapi.testclient import TestClient

from app.main import app


class FastAPIAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Explicitly entering the context manager triggers the app's
        # `lifespan` handler once for the whole test class — this is
        # what actually loads the preprocessor + the staging model from
        # DagsHub. Without this, requests would 500 with no model loaded.
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        # Matches templates/index.html's <title> exactly.
        self.assertIn(b"URLGuard", response.content)

    def test_predict_endpoint(self):
        # /predict expects JSON, not Flask-style form data — this app's
        # frontend (index.html) posts JSON via fetch(), not a <form>.
        # Hits a real URL (live page fetch for content-based features)
        # and the real staged model — this is an integration test, same
        # network-dependent style as test_model.py, not a pure unit test.
        response = self.client.post("/predict", json={"url": "https://google.com"})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn(data["prediction"], ("Safe", "Suspicious", "Phishing"))

    def test_predict_rejects_empty_url(self):
        response = self.client.post("/predict", json={"url": ""})
        self.assertEqual(response.status_code, 400)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])


if __name__ == "__main__":
    unittest.main()