# -*- coding: utf-8 -*-
"""
Agent模块

提供对话系统的主Agent类和消息处理器。
"""

from customer_agent.agent.agent import Agent, AgentConfig
from customer_agent.agent.message_processor import MessageProcessor
from customer_agent.agent.actions import (
    Action,
    ActionResult,
    ActionListen,
    ActionRestart,
    ActionDefaultFallback,
    ActionSendText,
)

__all__ = [
    "Agent",
    "AgentConfig",
    "MessageProcessor",
    "Action",
    "ActionResult",
    "ActionListen",
    "ActionRestart",
    "ActionDefaultFallback",
    "ActionSendText",
]
