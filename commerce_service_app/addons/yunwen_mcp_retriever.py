# -*- coding: utf-8 -*-
"""
Retriever adapter for querying the standalone yunwen MCP service.

This module is loaded from the commerce_service_app model directory by:
    python -m customer_agent run --model commerce_service_app
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from customer_agent.retrieval.base_retriever import InformationRetrieval, SearchResult

logger = logging.getLogger(__name__)

DEFAULT_MCP_URL = "http://127.0.0.1:9100/mcp"
DEFAULT_QUERY_TOOL = "yunwen_query_knowledge_base"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_SESSION_ID = "customer_agent_platform"


class YunwenMcpRetriever(InformationRetrieval):
    """InformationRetrieval implementation backed by the yunwen MCP service."""

    def __init__(self, embeddings=None, client_factory: Any = None) -> None:
        super().__init__(embeddings)
        self.url = DEFAULT_MCP_URL
        self.tool = DEFAULT_QUERY_TOOL
        self.timeout = DEFAULT_TIMEOUT_SECONDS
        self.enable_routing = True
        self.enable_crag = True
        self._client_factory = client_factory

    def connect(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Load MCP connection settings from endpoints.yml vector_store config."""
        config = config or {}
        self.url = config.get("url") or config.get("mcp_url") or self.url
        self.tool = config.get("tool") or config.get("query_tool") or self.tool
        self.timeout = int(config.get("timeout") or self.timeout)
        self.enable_routing = bool(config.get("enable_routing", self.enable_routing))
        self.enable_crag = bool(config.get("enable_crag", self.enable_crag))
        logger.info("YunwenMcpRetriever connected to %s using tool %s", self.url, self.tool)

    def _make_client(self):
        if self._client_factory is not None:
            return self._client_factory(self.url, timeout=self.timeout)

        from fastmcp import Client

        return Client(self.url, timeout=self.timeout)

    def _session_id_from_tracker(self, tracker_state: Optional[Dict[str, Any]]) -> str:
        if not tracker_state:
            return DEFAULT_SESSION_ID

        sender_id = tracker_state.get("sender_id")
        if sender_id:
            return str(sender_id)

        latest_message = tracker_state.get("latest_message")
        if isinstance(latest_message, dict) and latest_message.get("sender_id"):
            return str(latest_message["sender_id"])

        return DEFAULT_SESSION_ID

    def _payload_from_content(self, content: Any) -> Dict[str, Any]:
        if isinstance(content, dict):
            return content

        if isinstance(content, list):
            if not content:
                return {}
            content = content[0]

        text = getattr(content, "text", None)
        if text is None:
            text = str(content) if content is not None else ""

        if not text:
            return {}

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("yunwen MCP returned non-JSON text: %s", text[:300])
            return {"success": True, "answer": text}

        return payload if isinstance(payload, dict) else {}

    def _build_search_result(self, payload: Dict[str, Any]) -> Optional[SearchResult]:
        if not payload.get("success"):
            logger.warning("yunwen MCP query failed: %s", payload.get("error", "unknown error"))
            return None

        answer = (payload.get("answer") or "").strip()
        if not answer:
            logger.info("yunwen MCP returned an empty answer")
            return None

        metadata = {
            "source": "yunwen_mcp",
            "session_id": payload.get("session_id", ""),
            "citations": payload.get("citations", []),
            "image_urls": payload.get("image_urls", []),
            "query_type": payload.get("query_type", ""),
            "crag_decision": payload.get("crag_decision", ""),
        }
        return SearchResult(text=answer, metadata=metadata, score=1.0)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        tracker_state: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Query yunwen over MCP and map the final answer to SearchResult."""
        query = (query or "").strip()
        if not query:
            return []

        arguments = {
            "query": query,
            "session_id": self._session_id_from_tracker(tracker_state),
            "enable_routing": self.enable_routing,
            "enable_crag": self.enable_crag,
        }

        try:
            async with self._make_client() as client:
                content = await client.call_tool(
                    self.tool,
                    arguments,
                    timeout=self.timeout,
                )
        except Exception as exc:
            logger.error("yunwen MCP query error: %s", exc)
            return []

        payload = self._payload_from_content(content)
        result = self._build_search_result(payload)
        return [result] if result else []


__all__ = ["YunwenMcpRetriever"]
