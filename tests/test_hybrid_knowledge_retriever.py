# -*- coding: utf-8 -*-

import unittest

from customer_agent.retrieval.base_retriever import SearchResult
from commerce_service_app.addons.hybrid_knowledge_retriever import HybridKnowledgeRetriever


class RecordingRetriever:
    def __init__(self, name, response=None, error=None):
        self.name = name
        self.response = response if response is not None else [
            SearchResult(text=f"{name} answer", metadata={"source": name}, score=1.0)
        ]
        self.error = error
        self.queries = []

    def connect(self, config=None):
        self.config = config or {}

    async def search(self, query, top_k=5, tracker_state=None):
        self.queries.append(
            {"query": query, "top_k": top_k, "tracker_state": tracker_state}
        )
        if self.error:
            raise self.error
        return self.response


class HybridKnowledgeRetrieverTests(unittest.IsolatedAsyncioTestCase):
    async def test_recommendation_routes_to_product_retriever_not_yunwen(self):
        product = RecordingRetriever("product")
        yunwen = RecordingRetriever("yunwen")
        retriever = HybridKnowledgeRetriever(
            product_retriever=product,
            kb_retriever=yunwen,
        )

        results = await retriever.search("推荐一下手机", top_k=3)

        self.assertEqual("product answer", results[0].text)
        self.assertEqual(["推荐一下手机"], [call["query"] for call in product.queries])
        self.assertEqual([], yunwen.queries)

    async def test_document_question_routes_to_yunwen_retriever(self):
        product = RecordingRetriever("product")
        yunwen = RecordingRetriever("yunwen")
        retriever = HybridKnowledgeRetriever(
            product_retriever=product,
            kb_retriever=yunwen,
        )

        results = await retriever.search("介绍一下华为擎云B730计算机", top_k=3)

        self.assertEqual("yunwen answer", results[0].text)
        self.assertEqual([], product.queries)
        self.assertEqual(
            ["介绍一下华为擎云B730计算机"],
            [call["query"] for call in yunwen.queries],
        )

    async def test_order_service_query_never_calls_yunwen(self):
        product = RecordingRetriever("product")
        yunwen = RecordingRetriever("yunwen")
        retriever = HybridKnowledgeRetriever(
            product_retriever=product,
            kb_retriever=yunwen,
        )

        results = await retriever.search("查询订单", top_k=3)

        self.assertEqual([], results)
        self.assertEqual([], product.queries)
        self.assertEqual([], yunwen.queries)

    async def test_product_retriever_error_does_not_fallback_to_yunwen(self):
        product = RecordingRetriever("product", error=RuntimeError("neo4j down"))
        yunwen = RecordingRetriever("yunwen")
        retriever = HybridKnowledgeRetriever(
            product_retriever=product,
            kb_retriever=yunwen,
        )

        results = await retriever.search("帮我推荐一款键盘", top_k=3)

        self.assertEqual([], results)
        self.assertEqual(["帮我推荐一款键盘"], [call["query"] for call in product.queries])
        self.assertEqual([], yunwen.queries)


if __name__ == "__main__":
    unittest.main()
