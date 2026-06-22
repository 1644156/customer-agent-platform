# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace

from commerce_service_app.addons.information_retrieval import GraphRAG, RouteItem
from customer_agent.retrieval.base_retriever import SearchResult


OPPO_SKU = (
    "OPPO【分期免息】Find X8手机 全网通5G双卡双待 AI拍照新机 "
    "Find X8浮光白 12GB+256GB 简配+全国联保+版本咨询客服"
)
COSMETICS_SKU = "欧莱雅复颜玻尿酸水光充盈面霜 个护化妆 护肤品"
FOOD_SKU = "三只松鼠每日坚果 食品饮料 零食大礼包"
MOUSE_SKU = "罗技G502 HERO 游戏鼠标 高DPI 电竞鼠标"
KEYBOARD_SKU = "京东京造 JZ990 机械键盘 游戏键盘 RGB 红轴"
DISPLAY_SKU = "AOC 27英寸 2K 高刷 电竞显示器"


class FakeDriver:
    def __init__(self):
        self.queries = []

    def execute_query(self, cypher, params=None):
        self.queries.append((cypher, params))
        cypher_lower = cypher.lower()

        if cypher_lower.startswith("explain"):
            raise Exception("Cannot use aggregation in ORDER BY")
        if "match (n:category1)" in cypher_lower:
            categories = {
                "手机": {"category1_name": "手机"},
                "个护化妆": {"category1_name": "个护化妆"},
                "食品饮料": {"category1_name": "食品饮料"},
            }
            node = categories.get((params or {}).get("entity"))
            return SimpleNamespace(records=[{"n": node}] if node else [])
        if "match (n:category3)" in cypher_lower:
            categories = {
                "鼠标": {"category3_name": "鼠标"},
                "键盘": {"category3_name": "键盘"},
                "游戏键盘": {"category3_name": "键盘"},
                "机械键盘": {"category3_name": "键盘"},
                "显示器": {"category3_name": "显示器"},
            }
            node = categories.get((params or {}).get("entity"))
            return SimpleNamespace(records=[{"n": node}] if node else [])

        user_id = (params or {}).get("user_id")
        user_id_text = (params or {}).get("user_id_text")
        has_known_user = user_id == 35 or user_id_text == "35"
        if "match (u:user" in cypher_lower and "sku:sku" in cypher_lower and has_known_user:
            return SimpleNamespace(records=[{
                "sku_name": OPPO_SKU,
                "sku_price": None,
                "sku_desc": "AI拍照 手机 12GB 256GB",
            }])
        if "match (sku:sku)" in cypher_lower and (params or {}).get("category_terms") == ["个护化妆"]:
            return SimpleNamespace(records=[{
                "sku_name": COSMETICS_SKU,
            }])
        if "match (sku:sku)" in cypher_lower and (params or {}).get("category_terms") == ["食品饮料"]:
            return SimpleNamespace(records=[{
                "sku_name": FOOD_SKU,
            }])
        if "match (sku:sku)" in cypher_lower and (params or {}).get("category_terms") == ["鼠标"]:
            return SimpleNamespace(records=[{
                "sku_name": MOUSE_SKU,
            }])
        if "match (sku:sku)" in cypher_lower and (params or {}).get("category_terms") == ["键盘"]:
            return SimpleNamespace(records=[{
                "sku_name": KEYBOARD_SKU,
            }])
        if "match (sku:sku)" in cypher_lower and (params or {}).get("category_terms") == ["显示器"]:
            return SimpleNamespace(records=[{
                "sku_name": DISPLAY_SKU,
            }])

        return SimpleNamespace(records=[])


class WrongCategoryDriver(FakeDriver):
    def execute_query(self, cypher, params=None):
        self.queries.append((cypher, params))
        if cypher.lower().startswith("explain"):
            return SimpleNamespace(records=[])
        if "match (n:category3)" in cypher.lower() and (params or {}).get("entity") in {"游戏鼠标", "鼠标"}:
            return SimpleNamespace(records=[{"n": {"category3_name": "鼠标"}}])
        if (params or {}).get("category_terms") == ["鼠标"]:
            return SimpleNamespace(records=[{
                "sku_name": "Apple iPhone 12 (A2404) 64GB 白色 支持移动联通电信5G 双卡双待手机",
            }])
        return SimpleNamespace(records=[])


class FakeGraphRAG(GraphRAG):
    def __init__(self):
        super().__init__()
        self.driver = FakeDriver()
        self.neo4j_schema = "fake schema"
        self.cypher_corrector = lambda cypher: cypher

    async def route_label(self, query):
        return [RouteItem(label="Category1", entity="手机")]

    async def node_retrieval(self, route_res, top_k):
        return {
            "User": [{"user_id": 35}],
            "Category1": [{"category1_name": "手机"}],
        }

    async def generate_cypher(self, query, entry_nodes):
        return """MATCH (u:User {user_id: 35})-[:View]->(sku:SKU)-[:Belong]->(spu:SPU)-[:Belong]->(c3:Category3 {category3_name: '手机'})
WHERE EXISTS((spu)-[:Belong]->(:Trademark))
RETURN DISTINCT spu.spu_name
ORDER BY COUNT(sku) DESC
LIMIT 10"""

    async def validate_cypher(self, query, entry_nodes, cypher):
        return ["Cannot use aggregation in ORDER BY"]

    async def correct_cypher(self, query, entry_nodes, cypher, errors):
        return """MATCH (u:User {user_id: 35})-[:View]->(sku:SKU)-[:Belong]->(spu:SPU)-[:Belong]->(c3:Category3 {category3_name: '手机'})
WHERE EXISTS((spu)-[:Belong]->(:Trademark))
WITH spu, COUNT(sku) AS view_count
RETURN DISTINCT spu.spu_name
ORDER BY view_count DESC
LIMIT 10"""


class GraphRAGFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_repaired_cypher_uses_user_product_fallback(self):
        rag = FakeGraphRAG()

        results = await rag.search(
            "推荐一下手机，拍照用的",
            top_k=3,
            tracker_state={"slots": {"user_id": {"value": "35"}}, "events": []},
        )

        self.assertEqual(1, len(results))
        self.assertIn("OPPO", results[0].text)
        self.assertIn("AI拍照", results[0].text)

        executed_queries = [
            query
            for query, _ in rag.driver.queries
            if "sku:SKU" in query and "$category_terms" in query
        ]
        self.assertEqual(1, len(executed_queries))
        self.assertIn("$category_terms", executed_queries[0])


    async def test_repair_rejects_changed_user_id(self):
        rag = FakeGraphRAG()

        repaired = await rag._repair_cypher_until_valid(
            query="推荐一下化妆品",
            entry_nodes={"User": [{"user_id": 1}]},
            cypher="MATCH (u:User {user_id: 1}) RETURN u",
            errors=["user_id range mismatch"],
            max_attempts=1,
        )

        self.assertEqual("", repaired)

    async def test_cosmetics_fallback_uses_category_global_query_after_user_miss(self):
        rag = FakeGraphRAG()

        results = await rag._fallback_product_search(
            query="推荐一下化妆品",
            user_id="1",
            top_k=3,
            entry_nodes={"Category1": [{"category1_name": "个护化妆"}]},
        )

        self.assertEqual(1, len(results))
        self.assertIn("个护化妆", results[0].text)
        executed_queries = [
            query
            for query, _ in rag.driver.queries
            if "sku:SKU" in query
        ]
        self.assertEqual(3, len(executed_queries))
        self.assertTrue(all("MATCH (u:User)" in query for query in executed_queries[:2]))
        self.assertIn("MATCH (sku:SKU)", executed_queries[-1])
        self.assertIn("$category_terms", executed_queries[-1])
        self.assertIn("(c3:Category3)-[:Belong]->(c2:Category2)-[:Belong]->(c1:Category1)", executed_queries[-1])
        self.assertIn("c1.category1_name = term", executed_queries[-1])
        self.assertTrue(all("Buy" not in query and "Collect" not in query for query in executed_queries))
        self.assertTrue(all("sku_desc" not in query and "sku_price" not in query for query in executed_queries))

    async def test_food_query_maps_to_food_category_global_fallback(self):
        rag = FakeGraphRAG()

        results = await rag._fallback_product_search(
            query="推荐一下好吃的",
            user_id="1",
            top_k=3,
            entry_nodes={"Category1": [{"category1_name": "食品饮料"}]},
        )

        self.assertEqual(1, len(results))
        self.assertIn("食品饮料", results[0].text)
        executed_queries = [
            query
            for query, _ in rag.driver.queries
            if "sku:SKU" in query
        ]
        self.assertEqual(3, len(executed_queries))
        self.assertIn("MATCH (sku:SKU)", executed_queries[-1])
        _, params = next(
            (query, params)
            for query, params in reversed(rag.driver.queries)
            if params and "category_terms" in params
        )
        self.assertEqual(["食品饮料"], params["category_terms"])

    async def test_gaming_mouse_does_not_fallback_to_phone_category(self):
        rag = FakeGraphRAG()

        results = await rag._fallback_product_search(
            query="推荐一下游戏鼠标",
            user_id="1",
            top_k=3,
            entry_nodes={"Category3": [{"category3_name": "游戏本"}]},
        )

        self.assertEqual(1, len(results))
        self.assertIn("游戏鼠标", results[0].text)
        executed_queries = [
            query
            for query, _ in rag.driver.queries
            if "sku:SKU" in query
        ]
        self.assertEqual(3, len(executed_queries))
        _, params = next(
            (query, params)
            for query, params in reversed(rag.driver.queries)
            if params and "category_terms" in params
        )
        self.assertEqual(["鼠标"], params["category_terms"])
        self.assertNotIn("手机", params["category_terms"])
        self.assertNotIn("游戏本", params["category_terms"])

    async def test_keyboard_query_maps_to_keyboard_category(self):
        rag = FakeGraphRAG()

        results = await rag._fallback_product_search(
            query="推荐一下键盘",
            user_id="1",
            top_k=3,
            entry_nodes={},
        )

        self.assertEqual(1, len(results))
        self.assertIn("机械键盘", results[0].text)
        _, params = next(
            (query, params)
            for query, params in reversed(rag.driver.queries)
            if params and "category_terms" in params
        )
        self.assertEqual(["键盘"], params["category_terms"])

    async def test_unknown_code_category_can_be_inferred_from_graph(self):
        rag = FakeGraphRAG()

        results = await rag._fallback_product_search(
            query="推荐一下显示器",
            user_id="1",
            top_k=3,
            entry_nodes={},
        )

        self.assertEqual(1, len(results))
        self.assertIn("显示器", results[0].text)
        _, params = next(
            (query, params)
            for query, params in reversed(rag.driver.queries)
            if params and "category_terms" in params
        )
        self.assertEqual(["显示器"], params["category_terms"])

    async def test_wrong_category_fallback_result_is_rejected(self):
        rag = FakeGraphRAG()
        rag.driver = WrongCategoryDriver()

        results = await rag._fallback_product_search(
            query="推荐一下游戏鼠标",
            user_id="1",
            top_k=3,
            entry_nodes={},
        )

        self.assertEqual([], results)

    async def test_recommendation_without_category_rejects_broad_product_result(self):
        rag = FakeGraphRAG()

        results = rag._filter_semantic_results(
            query="推荐一下好用的",
            results=[SearchResult(text=f"商品名称: {OPPO_SKU}", score=1.0)],
            entry_nodes={},
        )

        self.assertEqual([], results)


if __name__ == "__main__":
    unittest.main()
