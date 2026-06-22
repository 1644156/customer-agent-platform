# -*- coding: utf-8 -*-
"""
Route customer-service search queries to the correct backend.

The commerce_service_app model has two different knowledge sources:
- yunwen MCP: standalone document/KB Q&A.
- GraphRAG: local ecommerce product graph for recommendation and shopping.

This adapter keeps EnterpriseSearchPolicy configured with one retriever while
preventing ecommerce intents from being sent to yunwen MCP.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Dict, List, Optional

from customer_agent.retrieval.base_retriever import InformationRetrieval, SearchResult

logger = logging.getLogger(__name__)

DEFAULT_KB_RETRIEVER = "addons.yunwen_mcp_retriever.YunwenMcpRetriever"
DEFAULT_PRODUCT_RETRIEVER = "addons.information_retrieval.GraphRAG"


class HybridKnowledgeRetriever(InformationRetrieval):
    """InformationRetrieval router for yunwen KB and local ecommerce GraphRAG."""

    PRODUCT_INTENT_KEYWORDS = (
        "推荐", "买什么", "哪个好", "哪款", "适合", "帮我选", "给我选",
        "有什么", "有哪些", "想买", "想要", "选购", "导购", "商品",
    )
    PRODUCT_CATEGORY_KEYWORDS = (
        "手机", "电脑", "笔记本", "键盘", "鼠标", "显示器", "台灯",
        "化妆品", "护肤", "美妆", "食品", "好吃",
    )
    ORDER_OR_FLOW_KEYWORDS = (
        "订单", "物流", "快递", "收货", "地址", "取消订单", "退货", "退款",
        "售后", "发票", "支付", "付款", "配送", "查询订单", "查订单",
    )

    def __init__(
        self,
        embeddings=None,
        product_retriever: Optional[InformationRetrieval] = None,
        kb_retriever: Optional[InformationRetrieval] = None,
    ) -> None:
        super().__init__(embeddings)
        self.product_retriever = product_retriever
        self.kb_retriever = kb_retriever

    def connect(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Create both retrievers from endpoints.yml vector_store config."""
        config = config or {}

        kb_config = dict(config.get("kb") or config.get("yunwen") or {})
        if not kb_config:
            kb_config = {
                key: value
                for key, value in config.items()
                if key not in {"product", "kb", "yunwen", "product_retriever", "kb_retriever"}
            }

        product_config = dict(config.get("product") or {})
        if not product_config:
            product_config = {
                "uri": os.getenv("NEO4J_URL", "bolt://127.0.0.1:7687"),
                "user": os.getenv("NEO4J_USER", "neo4j"),
                "password": os.getenv("NEO4J_PASSWORD", ""),
            }

        kb_class_path = config.get("kb_retriever") or DEFAULT_KB_RETRIEVER
        product_class_path = config.get("product_retriever") or DEFAULT_PRODUCT_RETRIEVER

        if self.kb_retriever is None:
            self.kb_retriever = self._create_retriever(kb_class_path, kb_config)
        elif kb_config:
            self.kb_retriever.connect(kb_config)

        if self.product_retriever is None:
            self.product_retriever = self._create_retriever(
                product_class_path,
                product_config,
                required=False,
            )
        elif product_config:
            self.product_retriever.connect(product_config)

        logger.info(
            "HybridKnowledgeRetriever connected: kb=%s, product=%s",
            type(self.kb_retriever).__name__ if self.kb_retriever else None,
            type(self.product_retriever).__name__ if self.product_retriever else None,
        )

    def _create_retriever(
        self,
        class_path: str,
        config: Dict[str, Any],
        required: bool = True,
    ) -> Optional[InformationRetrieval]:
        try:
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            retriever_class = getattr(module, class_name)
            retriever = retriever_class()
            retriever.connect(config)
            return retriever
        except Exception as exc:
            if required:
                raise
            logger.warning("Optional retriever %s unavailable: %s", class_path, exc)
            return None

    def _is_order_or_flow_query(self, query: str) -> bool:
        return any(keyword in query for keyword in self.ORDER_OR_FLOW_KEYWORDS)

    def _is_product_query(self, query: str) -> bool:
        has_product_intent = any(keyword in query for keyword in self.PRODUCT_INTENT_KEYWORDS)
        has_product_category = any(keyword in query for keyword in self.PRODUCT_CATEGORY_KEYWORDS)
        return has_product_intent or ("商品" in query and has_product_category)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        tracker_state: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        query = (query or "").strip()
        if not query:
            return []

        if self._is_order_or_flow_query(query):
            logger.info("HybridKnowledgeRetriever skip retrieval for service flow query: %s", query)
            return []

        if self._is_product_query(query):
            if not self.product_retriever:
                logger.warning("Product query skipped because GraphRAG retriever is unavailable: %s", query)
                return []
            try:
                logger.info("HybridKnowledgeRetriever route=product query=%s", query)
                return await self.product_retriever.search(
                    query,
                    top_k=top_k,
                    tracker_state=tracker_state,
                )
            except Exception as exc:
                logger.error("Product retriever error: %s", exc)
                return []

        if not self.kb_retriever:
            logger.warning("KB query skipped because yunwen retriever is unavailable: %s", query)
            return []

        try:
            logger.info("HybridKnowledgeRetriever route=yunwen query=%s", query)
            return await self.kb_retriever.search(
                query,
                top_k=top_k,
                tracker_state=tracker_state,
            )
        except Exception as exc:
            logger.error("KB retriever error: %s", exc)
            return []

    def close(self) -> None:
        for retriever in (self.product_retriever, self.kb_retriever):
            close = getattr(retriever, "close", None)
            if callable(close):
                close()


__all__ = ["HybridKnowledgeRetriever"]
