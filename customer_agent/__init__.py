# -*- coding: utf-8 -*-
"""
customer_agent - 任务型客服 Agent 平台

基于 LLM、LangGraph、YAML Flow 和业务 Action 的客服 Agent 后端原型。
项目重点展示多轮状态管理、可控业务流程、知识源路由和工程化测试能力。

核心模块：
- agent: 对话代理，基于LangGraph实现图式消息处理流程
- dialogue_understanding: 对话理解模块(DU)，包含命令生成、处理、Flow执行
- core: 核心对话管理，包含Tracker、Domain、Slot、Store
- policies: 对话策略，包含FlowPolicy、EnterpriseSearchPolicy
- nlg: 自然语言生成（预留）
- retrieval: 检索增强(FAISS向量检索)（预留）
- training: 训练模块
- api: Web服务(FastAPI)
- channels: 对话通道（REST、SocketIO、Console）
- shared: 共享工具和配置
"""

__version__ = "0.1.0"
__author__ = "customer_agent"

from customer_agent.shared.constants import (
    DEFAULT_SERVER_PORT,
    DEFAULT_MODELS_PATH,
)

__all__ = [
    "__version__",
    "DEFAULT_SERVER_PORT",
    "DEFAULT_MODELS_PATH",
]
