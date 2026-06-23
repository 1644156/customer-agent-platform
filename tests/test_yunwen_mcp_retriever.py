# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace

from commerce_service_app.addons.yunwen_mcp_retriever import YunwenMcpRetriever


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def call_tool(self, tool, arguments, timeout=None):
        self.calls.append(
            {
                "tool": tool,
                "arguments": arguments,
                "timeout": timeout,
            }
        )
        if self.error:
            raise self.error
        return self.response


class ClientFactory:
    def __init__(self, client):
        self.client = client
        self.calls = []

    def __call__(self, url, timeout=None):
        self.calls.append({"url": url, "timeout": timeout})
        return self.client


class YunwenMcpRetrieverTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_maps_success_payload_to_search_result(self):
        payload = {
            "success": True,
            "answer": "HAK 180 烫金机需要先阅读说明书。",
            "session_id": "user-1",
            "citations": [{"title": "说明书"}],
            "image_urls": ["http://example.com/a.jpg"],
            "query_type": "factual",
            "crag_decision": "correct",
        }
        client = FakeClient(response=[SimpleNamespace(text=__import__("json").dumps(payload, ensure_ascii=False))])
        factory = ClientFactory(client)
        retriever = YunwenMcpRetriever(client_factory=factory)
        retriever.connect({"url": "http://127.0.0.1:9100/mcp", "tool": "yunwen_query_knowledge_base", "timeout": 12})

        results = await retriever.search(
            "HAK 180 怎么使用？",
            tracker_state={"sender_id": "user-1"},
        )

        self.assertEqual(1, len(results))
        self.assertIn("HAK 180", results[0].text)
        self.assertEqual("yunwen_mcp", results[0].metadata["source"])
        self.assertEqual([{"title": "说明书"}], results[0].metadata["citations"])
        self.assertEqual("user-1", client.calls[0]["arguments"]["session_id"])
        self.assertEqual(12, client.calls[0]["timeout"])

    async def test_search_uses_latest_message_sender_when_sender_missing(self):
        payload = {"success": True, "answer": "有答案", "session_id": "sender-from-message"}
        client = FakeClient(response=[SimpleNamespace(text=__import__("json").dumps(payload, ensure_ascii=False))])
        retriever = YunwenMcpRetriever(client_factory=ClientFactory(client))

        await retriever.search(
            "问题",
            tracker_state={"latest_message": {"sender_id": "sender-from-message"}},
        )

        self.assertEqual("sender-from-message", client.calls[0]["arguments"]["session_id"])

    async def test_search_returns_empty_on_failed_payload(self):
        payload = {"success": False, "error": "timeout"}
        client = FakeClient(response=[SimpleNamespace(text=__import__("json").dumps(payload, ensure_ascii=False))])
        retriever = YunwenMcpRetriever(client_factory=ClientFactory(client))

        results = await retriever.search("问题")

        self.assertEqual([], results)

    async def test_search_returns_empty_on_empty_answer(self):
        payload = {"success": True, "answer": ""}
        client = FakeClient(response=[SimpleNamespace(text=__import__("json").dumps(payload, ensure_ascii=False))])
        retriever = YunwenMcpRetriever(client_factory=ClientFactory(client))

        results = await retriever.search("问题")

        self.assertEqual([], results)

    async def test_search_returns_empty_on_client_error(self):
        client = FakeClient(error=RuntimeError("connection refused"))
        retriever = YunwenMcpRetriever(client_factory=ClientFactory(client))

        results = await retriever.search("问题")

        self.assertEqual([], results)


if __name__ == "__main__":
    unittest.main()
