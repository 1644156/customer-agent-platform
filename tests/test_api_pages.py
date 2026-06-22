# -*- coding: utf-8 -*-

import unittest

from fastapi.testclient import TestClient

from customer_agent.api.server import create_app


class ApiPageTests(unittest.TestCase):
    def test_chat_page_is_available(self):
        client = TestClient(create_app(agent=None))

        response = client.get("/chat")

        self.assertEqual(200, response.status_code)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("智能客服", response.text)
        self.assertIn("/api/stream", response.text)
        self.assertIn("/api/tracker/", response.text)
        self.assertIn("/api/sessions/", response.text)
        self.assertIn("citations", response.text)
        self.assertIn("image_urls", response.text)
        self.assertIn("清空当前会话", response.text)
        self.assertIn("userIdInput", response.text)
        self.assertIn("/SetSlots(user_id=", response.text)
        self.assertIn("extractImageUrlsFromText", response.text)
        self.assertIn("stripImageBlock", response.text)
        self.assertIn("encodeURI", response.text)

    def test_inspect_page_still_available(self):
        client = TestClient(create_app(agent=None, enable_inspect=True))

        response = client.get("/inspect")

        self.assertEqual(200, response.status_code)
        self.assertIn("text/html", response.headers["content-type"])


if __name__ == "__main__":
    unittest.main()
