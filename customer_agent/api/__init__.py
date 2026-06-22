# -*- coding: utf-8 -*-
"""
customer_agent API模块

提供基于FastAPI的Web服务接口。
"""

from customer_agent.api.server import CustomerAgentServer, create_app

__all__ = [
    "CustomerAgentServer",
    "create_app",
]
