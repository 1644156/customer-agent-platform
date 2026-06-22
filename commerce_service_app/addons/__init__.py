# -*- coding: utf-8 -*-
"""
电商客服Demo Addons模块
"""

__all__ = ["GraphRAG", "YunwenMcpRetriever", "HybridKnowledgeRetriever"]


def __getattr__(name):
    if name == "GraphRAG":
        from .information_retrieval import GraphRAG

        return GraphRAG
    if name == "YunwenMcpRetriever":
        from .yunwen_mcp_retriever import YunwenMcpRetriever

        return YunwenMcpRetriever
    if name == "HybridKnowledgeRetriever":
        from .hybrid_knowledge_retriever import HybridKnowledgeRetriever

        return HybridKnowledgeRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
