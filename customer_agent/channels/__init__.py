# -*- coding: utf-8 -*-
"""
通道模块

提供与用户交互的通道（Channel）实现。
"""

from customer_agent.channels.base_channel import (
    InputChannel,
    OutputChannel,
    UserMessage,
    CollectingOutputChannel,
)
from customer_agent.channels.rest_channel import RestChannel
from customer_agent.channels.socketio_channel import SocketIOChannel
from customer_agent.channels.console_channel import ConsoleChannel
from customer_agent.channels.inspect_proxy import InspectProxy

__all__ = [
    "InputChannel",
    "OutputChannel",
    "UserMessage",
    "CollectingOutputChannel",
    "RestChannel",
    "SocketIOChannel",
    "ConsoleChannel",
    "InspectProxy",
]
