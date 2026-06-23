# -*- coding: utf-8 -*-
"""
NLG模块（自然语言生成）

负责生成机器人的回复文本。
"""

from customer_agent.nlg.nlg_generator import NLGGenerator, NLGConfig, NLGResponse
from customer_agent.nlg.template_nlg import TemplateNLG
from customer_agent.nlg.response_rephraser import ResponseRephraser, RephraserConfig

__all__ = [
    "NLGGenerator",
    "NLGConfig",
    "NLGResponse",
    "TemplateNLG",
    "ResponseRephraser",
    "RephraserConfig",
]
