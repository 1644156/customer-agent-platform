# -*- coding: utf-8 -*-
"""
企业搜索策略

基于知识库检索的策略，实现RAG功能和降级机制。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from customer_agent.policies.base_policy import Policy, PolicyConfig, PolicyPrediction
from customer_agent.shared.constants import DegradationReason, ACTION_DEFAULT_FALLBACK
from customer_agent.shared.llm import create_llm_client
from customer_agent.shared.llm.base_client import LLMClient
from customer_agent.retrieval.base_retriever import SearchResult

if TYPE_CHECKING:
    from customer_agent.core.tracker import DialogueStateTracker
    from customer_agent.core.domain import Domain
    from customer_agent.dialogue_understanding.flow import FlowsList
    from customer_agent.dialogue_understanding.stack.stack_frame import StackFrame

logger = logging.getLogger(__name__)


@dataclass
class _InternalRetrievalConfig:
    """内部检索配置（简化版）。"""
    enabled: bool = True
    top_k: int = 3
    similarity_threshold: float = 0.5


@dataclass
class EnterpriseSearchPolicyConfig(PolicyConfig):
    """企业搜索策略配置。
    
    Attributes:
        priority: 策略优先级
        retrieval: 检索配置
        llm_type: LLM类型 (openai/qwen/azure/anthropic)
        llm_model: LLM模型名
        enable_citation: 是否启用引用
        enable_relevancy_check: 是否启用相关性检查
        chitchat_enabled: 是否启用闲聊降级
    """
    priority: int = 50  # 中等优先级
    retrieval: _InternalRetrievalConfig = field(default_factory=_InternalRetrievalConfig)
    llm_type: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    enable_citation: bool = False
    enable_relevancy_check: bool = True
    chitchat_enabled: bool = True


class EnterpriseSearchPolicy(Policy):
    """企业搜索策略。
    
    基于知识库检索实现RAG功能，并包含内置的降级机制。
    
    降级链：
    1. Flow匹配 → 执行Flow
    2. 知识库检索 → 生成RAG回答
    3. 闲聊栈帧 → 生成闲聊回复
    4. 无法处理 → 返回默认回复
    
    工作流程：
    1. 检索相关文档
    2. 检查相关性
    3. 使用LLM生成回答
    4. 如果检索无结果，按配置降级到闲聊或默认回复
    """
    
    DEFAULT_PRIORITY = 50
    NON_KB_RECOMMENDATION_SLOT = "__non_kb_recommendation_context"
    PRODUCT_PREFERENCE_KEYWORDS = (
        "打游戏", "游戏", "电竞", "办公", "拍照", "摄影", "预算", "以内",
        "内吧", "元", "块", "便宜", "贵", "性价比", "手感", "静音",
        "无线", "有线", "蓝牙", "机械", "青轴", "茶轴", "红轴", "黑轴",
        "学习", "护眼", "照明", "氛围", "工作", "宿舍", "卧室", "书桌",
        "尺寸", "功率", "品牌", "风格", "学生", "儿童", "老人",
        "颜色", "白色", "黑色", "轻薄", "便携", "续航", "快充",
        "长焦", "人像", "夜景", "低延迟", "防水", "静音", "护腕",
    )
    NON_KB_CONTEXT_CANCEL_KEYWORDS = (
        "谢谢", "不用", "算了", "不需要", "换个话题", "先这样", "没事",
        "取消", "停止",
    )
    
    # RAG提示词模板
    RAG_PROMPT_TEMPLATE = """你是一个专业的客服助手，正在根据知识库文档回答用户问题。

### 参考文档
{context}

### 用户问题
{question}

### 回答要求
严格基于上述文档内容回答：
1. 直接回答问题，不要添加问候语或寒暄
2. 禁止使用 emoji 表情符号
3. 使用专业、简洁的语气
4. 只陈述文档中明确提到的信息
5. 如果文档包含具体的产品名称、品牌、规格等，必须准确引用
6. 最多2-3句话，避免冗余
7. 如果文档信息不足以回答问题，仅回复"[NO_RAG_ANSWER]"

回答：
"""
    
    # 闲聊提示词模板
    CHITCHAT_PROMPT_TEMPLATE = """你是一个友好的AI助手。当前没有可用的业务流程或可靠知识库结果，需要自然降级回复用户。

回复必须符合逻辑：
1. 不要声称已经查到知识库、库存、用户历史或商品详情
2. 如果用户在要商品推荐，但没有可靠商品数据，不要编造具体SKU、型号或库存；可以给通用选购思路，并追问预算、品牌、用途等关键条件
3. 如果用户只是闲聊，就正常简短回应
4. 禁止使用 emoji 表情符号

用户: {message}

请回复（保持简短友好）：
"""
    
    NON_KB_RECOMMENDATION_PROMPT_TEMPLATE = """你是一个专业、谨慎的购物建议助手。

当前知识库没有可靠商品命中，请基于通用商品知识回答用户的推荐需求。

### 用户需求
{message}

### 回答要求
1. 可以给出选购方向、关键参数、适合的类型、价位建议和常见品牌方向
2. 不要声称这些商品来自知识库、库存或用户历史
3. 不要编造具体在售SKU、库存、价格或平台承诺
4. 如果信息仍不足，先给2-3条方向，再问一个最关键的补充问题
5. 禁止使用 emoji，回复简洁自然

回答：
"""
    
    def __init__(
        self,
        config: Optional[EnterpriseSearchPolicyConfig] = None,
        llm_client: Optional[LLMClient] = None,
        retriever: Optional[Any] = None,
        **kwargs: Any,
    ):
        """初始化企业搜索策略。
        
        Args:
            config: 策略配置
            llm_client: LLM客户端
            retriever: 检索器
            **kwargs: 额外参数
        """
        super().__init__(config or EnterpriseSearchPolicyConfig(), **kwargs)
        self.config: EnterpriseSearchPolicyConfig = self.config
        
        self._llm_client = llm_client
        self._retriever = retriever
    
    @property
    def llm_client(self) -> LLMClient:
        """获取LLM客户端（延迟初始化）。"""
        if self._llm_client is None:
            self._llm_client = create_llm_client(
                type=self.config.llm_type,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
            )
        return self._llm_client
    
    def does_support_stack_frame(self, frame: Optional[Any] = None) -> bool:
        """检查策略是否支持处理指定栈帧。
        
        支持：SearchStackFrame、ChitChatStackFrame、CannotHandleStackFrame、
              CompletedStackFrame、HumanHandoffStackFrame
        
        Args:
            frame: 要检查的栈帧
            
        Returns:
            是否支持处理该栈帧
        """
        from customer_agent.dialogue_understanding.stack.stack_frame import (
            SearchStackFrame,
            ChitChatStackFrame,
            CannotHandleStackFrame,
            CompletedStackFrame,
            HumanHandoffStackFrame,
        )
        return isinstance(frame, (
            SearchStackFrame, 
            ChitChatStackFrame, 
            CannotHandleStackFrame,
            CompletedStackFrame,
            HumanHandoffStackFrame,
        ))
    
    async def predict(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        flows: Optional["FlowsList"] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """预测下一步动作。
        
        检测栈帧类型并分发处理：
        - SearchStackFrame → 执行检索
        - ChitChatStackFrame → 生成闲聊回复
        - CannotHandleStackFrame → 返回降级响应
        - CompletedStackFrame → 询问是否还有其他需求
        - HumanHandoffStackFrame → 执行人工转接
        
        Args:
            tracker: 对话状态追踪器
            domain: Domain定义
            flows: Flow列表
            **kwargs: 额外参数
            
        Returns:
            预测结果
        """
        from customer_agent.dialogue_understanding.stack.stack_frame import (
            SearchStackFrame,
            ChitChatStackFrame,
            CannotHandleStackFrame,
            CompletedStackFrame,
            HumanHandoffStackFrame,
        )
        
        # 获取栈顶帧
        top_frame = tracker.dialogue_stack.top()
        
        # 检查是否已经有 bot 响应（如果刚执行了动作，则放弃）
        # 但是对于需要立即处理的栈帧（如 CompletedStackFrame），不应放弃
        from customer_agent.dialogue_understanding.stack.stack_frame import (
            CompletedStackFrame as CompletedFrame,
            HumanHandoffStackFrame as HandoffFrame,
        )
        needs_immediate_handling = isinstance(top_frame, (CompletedFrame, HandoffFrame))
        
        if (tracker.latest_action_name 
            and tracker.latest_action_name != "action_listen"
            and not needs_immediate_handling):
            logger.debug(f"Action {tracker.latest_action_name} just executed, abstaining")
            return PolicyPrediction.abstain(self.name)
        
        # 获取用户消息（统一从latest_message获取）
        user_message = ""
        if tracker.latest_message:
            user_message = tracker.latest_message.text
        
        # 根据栈帧类型分发处理
        if isinstance(top_frame, CompletedStackFrame):
            return await self._handle_completed_frame(tracker, top_frame, domain)
        
        if isinstance(top_frame, HumanHandoffStackFrame):
            return await self._handle_human_handoff_frame(tracker, top_frame, domain)
        
        if isinstance(top_frame, ChitChatStackFrame):
            return await self._handle_chitchat_frame(tracker, user_message)
        
        if isinstance(top_frame, CannotHandleStackFrame):
            return await self._handle_cannot_handle_frame(tracker, top_frame, domain)
        
        if isinstance(top_frame, SearchStackFrame):
            return await self._handle_search_frame(tracker, user_message)
        
        # 没有特定栈帧，放弃处理
        return PolicyPrediction.abstain(self.name)
    
    async def _handle_search_frame(
        self,
        tracker: "DialogueStateTracker",
        user_message: str,
    ) -> PolicyPrediction:
        """处理SearchStackFrame - 执行检索。"""
        if not user_message:
            return PolicyPrediction.abstain(self.name)

        non_kb_message = self._resolve_non_kb_recommendation_message(tracker, user_message)
        if non_kb_message:
            logger.info("[EnterpriseSearchPolicy] 使用非知识库推荐上下文回答: %s", non_kb_message)
            return await self._handle_non_kb_recommendation(tracker, non_kb_message, pattern="search")

        user_message = self._resolve_contextual_recommendation_query(tracker, user_message)
        
        logger.info(f"[EnterpriseSearchPolicy] SearchStackFrame processing: {user_message}")
        
        try:
            # 尝试知识库检索
            if self.config.retrieval.enabled and self._retriever:
                search_results = await self._search(user_message, tracker)
                
                if search_results:
                    direct_answer = self._build_direct_retrieval_prediction(search_results)
                    if direct_answer:
                        self._clear_non_kb_recommendation_context(tracker)
                        tracker.dialogue_stack.pop()
                        tracker.record_pattern("search")
                        logger.info(
                            "[EnterpriseSearchPolicy] using direct retrieval answer from %s",
                            direct_answer.metadata.get("source"),
                        )
                        logger.debug("SearchStackFrame popped after direct retrieval answer")
                        return direct_answer

                    logger.info(f"[EnterpriseSearchPolicy] 检索到 {len(search_results)} 条结果，开始生成RAG回答")
                    answer = await self._generate_rag_answer(user_message, search_results)
                    logger.info(f"[EnterpriseSearchPolicy] RAG回答: {answer[:200] if answer else 'None'}...")
                    
                    if answer and "[NO_RAG_ANSWER]" not in answer:
                        self._clear_non_kb_recommendation_context(tracker)
                        # 检索成功，弹出栈帧
                        tracker.dialogue_stack.pop()
                        # 记录 Pattern 执行历史
                        tracker.record_pattern("search")
                        logger.debug("SearchStackFrame popped after successful retrieval")
                        
                        return PolicyPrediction(
                            action="action_send_text",
                            confidence=0.9,
                            policy_name=self.name,
                            metadata={
                                "text": answer,
                                "degradation_reason": DegradationReason.DEFAULT,
                                "search_results": [r.content for r in search_results],
                            },
                        )

                    grounded_answer = self._build_grounded_result_answer(search_results)
                    self._clear_non_kb_recommendation_context(tracker)
                    tracker.dialogue_stack.pop()
                    tracker.record_pattern("search")
                    logger.debug("SearchStackFrame popped after grounded retrieval fallback")

                    return PolicyPrediction(
                        action="action_send_text",
                        confidence=0.75,
                        policy_name=self.name,
                        metadata={
                            "text": grounded_answer,
                            "degradation_reason": DegradationReason.NO_RELEVANT_ANSWER,
                            "search_results": [r.content for r in search_results],
                        },
                    )
            
            if self.config.chitchat_enabled:
                self._remember_non_kb_recommendation_context(tracker, user_message)
                chitchat_answer = await self._generate_chitchat_answer(user_message)
                if chitchat_answer:
                    chitchat_answer = self._sanitize_chitchat_answer(user_message, chitchat_answer)
                    tracker.dialogue_stack.pop()
                    tracker.record_pattern("search")
                    logger.debug("SearchStackFrame popped after chitchat fallback")

                    return PolicyPrediction(
                        action="action_send_text",
                        confidence=0.7,
                        policy_name=self.name,
                        metadata={
                            "text": chitchat_answer,
                            "degradation_reason": DegradationReason.CHITCHAT,
                        },
                    )

            tracker.dialogue_stack.pop()
            tracker.record_pattern("search")
            return PolicyPrediction(
                action=ACTION_DEFAULT_FALLBACK,
                confidence=0.5,
                policy_name=self.name,
                metadata={"degradation_reason": DegradationReason.CANNOT_HANDLE},
            )
            
        except Exception as e:
            logger.error(f"Search frame error: {e}")
            try:
                tracker.dialogue_stack.pop()
                # 记录 Pattern 执行历史
                tracker.record_pattern("search")
            except Exception:
                pass
            return PolicyPrediction(
                action=ACTION_DEFAULT_FALLBACK,
                confidence=0.3,
                policy_name=self.name,
                metadata={"degradation_reason": DegradationReason.INTERNAL_ERROR, "error": str(e)},
            )

    def _knowledge_no_result_text(self) -> str:
        """检索结果无法支撑回答时的保守回复。"""
        return "我这边没有拿到足够可靠的商品依据。你可以补充预算、品牌或使用场景，我再帮你缩小范围。"

    def _tracker_get_slot(self, tracker: "DialogueStateTracker", slot_name: str) -> Any:
        """兼容真实 Tracker 和测试替身读取内部策略槽位。"""
        if not tracker:
            return None
        if hasattr(tracker, "get_slot"):
            try:
                return tracker.get_slot(slot_name)
            except Exception:
                pass
        slots = getattr(tracker, "slots", None)
        if isinstance(slots, dict):
            slot = slots.get(slot_name)
            return getattr(slot, "value", slot)
        return None

    def _tracker_set_slot(self, tracker: "DialogueStateTracker", slot_name: str, value: Any) -> None:
        """兼容真实 Tracker 和测试替身写入内部策略槽位。"""
        if not tracker:
            return
        if hasattr(tracker, "set_slot"):
            try:
                tracker.set_slot(slot_name, value)
                return
            except Exception:
                pass
        slots = getattr(tracker, "slots", None)
        if isinstance(slots, dict):
            slots[slot_name] = value

    def _remember_non_kb_recommendation_context(
            self,
            tracker: "DialogueStateTracker",
            query: str,
    ) -> None:
        """记录知识库无命中的推荐对象，供后续补充条件直接走通用推荐。"""
        if not self._is_recommendation_query(query):
            return
        context = {
            "query": query,
            "reason": "empty_retrieval",
        }
        self._tracker_set_slot(tracker, self.NON_KB_RECOMMENDATION_SLOT, context)

    def _clear_non_kb_recommendation_context(self, tracker: "DialogueStateTracker") -> None:
        """清理非知识库推荐上下文。"""
        self._tracker_set_slot(tracker, self.NON_KB_RECOMMENDATION_SLOT, None)

    def _get_non_kb_recommendation_context(self, tracker: "DialogueStateTracker") -> Optional[dict]:
        """获取非知识库推荐上下文。"""
        context = self._tracker_get_slot(tracker, self.NON_KB_RECOMMENDATION_SLOT)
        return context if isinstance(context, dict) and context.get("query") else None

    def _is_recommendation_query(self, message: str) -> bool:
        """判断用户是否在请求商品推荐。"""
        message = message or ""
        keywords = (
            "推荐", "买什么", "哪个好", "哪款", "适合", "帮我选", "给我选",
            "有什么", "有哪些", "想买", "想要",
        )
        return any(keyword in message for keyword in keywords)

    def _looks_like_product_preference_followup(self, message: str) -> bool:
        """识别“用途/预算/偏好”这类承接上一轮推荐的补充句。"""
        message = message or ""
        return any(keyword in message for keyword in self.PRODUCT_PREFERENCE_KEYWORDS)

    def _looks_like_non_kb_recommendation_followup(self, message: str) -> bool:
        """识别知识库无命中后的补充条件，避免重新检索出错类目。"""
        message = (message or "").strip()
        if not message or self._is_recommendation_query(message):
            return False
        if any(keyword in message for keyword in self.NON_KB_CONTEXT_CANCEL_KEYWORDS):
            return False
        return self._looks_like_product_preference_followup(message)

    def _resolve_non_kb_recommendation_message(
            self,
            tracker: "DialogueStateTracker",
            user_message: str,
    ) -> Optional[str]:
        """如果上一轮推荐已确认知识库无命中，将本轮补充拼成通用推荐需求。"""
        context = self._get_non_kb_recommendation_context(tracker)
        if not context:
            return None
        if not self._looks_like_non_kb_recommendation_followup(user_message):
            return None
        return f"{context['query']}；补充条件：{user_message}"

    def _last_recommendation_query_from_tracker(self, tracker: "DialogueStateTracker") -> Optional[str]:
        """从历史中找最近一条推荐请求，用于承接预算/用途等补充条件。"""
        if not tracker or not hasattr(tracker, "get_messages_for_llm"):
            return None

        try:
            messages = tracker.get_messages_for_llm(max_turns=5)
        except Exception:
            return None

        for message in reversed(messages[:-1]):
            if message.get("role") != "user":
                continue
            content = (message.get("content") or "").strip()
            if self._is_recommendation_query(content):
                return content
        return None

    def _resolve_contextual_recommendation_query(
            self,
            tracker: "DialogueStateTracker",
            user_message: str,
    ) -> str:
        """把承接上一轮推荐的补充条件拼回完整查询。"""
        if self._is_recommendation_query(user_message):
            return user_message
        if not self._looks_like_product_preference_followup(user_message):
            return user_message

        previous_query = self._last_recommendation_query_from_tracker(tracker)
        if not previous_query:
            return user_message

        resolved = f"{previous_query}；补充条件：{user_message}"
        logger.info("[EnterpriseSearchPolicy] 补全上下文推荐查询: %s", resolved)
        return resolved

    def _should_route_chitchat_to_search(
            self,
            tracker: "DialogueStateTracker",
            user_message: str,
    ) -> bool:
        """命令生成器误判补充条件为闲聊时，重新导向检索。"""
        return (
            not self._is_recommendation_query(user_message)
            and self._looks_like_product_preference_followup(user_message)
            and self._last_recommendation_query_from_tracker(tracker) is not None
        )

    def _looks_like_specific_product_recommendation(self, answer: str) -> bool:
        """识别无检索依据时不应出现的具体商品/型号推荐。"""
        if not answer:
            return False
        product_signals = (
            "iphone", "ipad", "oppo", "vivo", "小米", "红米", "华为", "荣耀",
            "一加", "三星", "find x", "mate", "nova", "罗技", "雷蛇", "g502",
            "欧莱雅", "兰蔻", "雅诗兰黛", "三只松鼠",
        )
        answer_lower = answer.lower()
        if any(signal in answer_lower for signal in product_signals):
            return True
        return bool(re.search(r"[A-Za-z]+[ -]?\d{1,3}\s*(?:pro|max|plus|ultra)?", answer, re.IGNORECASE))

    def _sanitize_chitchat_answer(self, user_message: str, answer: str) -> str:
        """无可靠检索结果时，阻止降级闲聊编造具体商品。"""
        if (
            self._is_recommendation_query(user_message)
            and self._looks_like_specific_product_recommendation(answer)
        ):
            logger.warning("降级闲聊疑似编造具体商品推荐，改用通用选购建议")
            return self._generic_recommendation_fallback_text(user_message)
        return answer

    def _generic_recommendation_fallback_text(self, message: str) -> str:
        """在没有可靠商品依据或 LLM 输出不可用时，给出自然的通用推荐方向。"""
        message = (message or "").strip()
        if "；补充条件：" in message:
            base, preference = message.split("；补充条件：", 1)
            preference = preference.strip()
            if preference:
                return (
                    f"可以按“{preference}”这个场景先筛选：优先看和用途直接相关的核心参数，"
                    "再结合预算、售后、真实评价和使用环境取舍。"
                    "你给我一个预算范围，我可以继续帮你把选择方向收窄。"
                )
            message = base.strip()

        return (
            "可以先按用途、预算和关键参数来筛选，不直接给没有依据的具体型号。"
            "如果你补充主要使用场景、预算或偏好的品牌/规格，我可以继续帮你缩小范围。"
        )

    def _build_grounded_result_answer(self, search_results: List[SearchResult]) -> str:
        """RAG 生成失败时，直接返回检索到的事实片段。"""
        snippets = []
        for result in search_results[:3]:
            content = " ".join((result.content or "").split())
            if not content:
                continue
            if len(content) > 240:
                content = content[:237] + "..."
            snippets.append(content)

        if not snippets:
            return self._knowledge_no_result_text()

        lines = [f"{idx}. {content}" for idx, content in enumerate(snippets, 1)]
        return "检索到以下相关信息：\n" + "\n".join(lines)

    def _build_direct_retrieval_prediction(
        self,
        search_results: List[SearchResult],
    ) -> Optional[PolicyPrediction]:
        """Return final answers from retrievers that already perform generation."""
        if not search_results:
            return None

        result = search_results[0]
        metadata = result.metadata or {}
        if metadata.get("source") != "yunwen_mcp":
            return None

        answer = (result.content or "").strip()
        if not answer:
            return None

        return PolicyPrediction(
            action="action_send_text",
            confidence=0.9,
            policy_name=self.name,
            metadata={
                "text": answer,
                "degradation_reason": DegradationReason.DEFAULT,
                "search_results": [r.content for r in search_results],
                "source": "yunwen_mcp",
                "citations": metadata.get("citations", []),
                "image_urls": metadata.get("image_urls", []),
                "session_id": metadata.get("session_id", ""),
                "query_type": metadata.get("query_type", ""),
                "crag_decision": metadata.get("crag_decision", ""),
            },
        )

    async def _handle_non_kb_recommendation(
            self,
            tracker: "DialogueStateTracker",
            message: str,
            pattern: str,
    ) -> PolicyPrediction:
        """知识库无命中后的通用推荐生成，不再重新检索。"""
        answer = await self._generate_non_kb_recommendation_answer(message)
        if not answer:
            answer = self._generic_recommendation_fallback_text(message)

        self._tracker_set_slot(
            tracker,
            self.NON_KB_RECOMMENDATION_SLOT,
            {"query": message, "reason": "non_kb_followup"},
        )
        tracker.dialogue_stack.pop()
        tracker.record_pattern(pattern)

        return PolicyPrediction(
            action="action_send_text",
            confidence=0.68,
            policy_name=self.name,
            metadata={
                "text": answer,
                "degradation_reason": DegradationReason.CHITCHAT,
                "non_kb_recommendation": True,
            },
        )
    
    async def _handle_chitchat_frame(
        self,
        tracker: "DialogueStateTracker",
        user_message: str,
    ) -> PolicyPrediction:
        """处理ChitChatStackFrame - 生成闲聊回复。"""
        logger.debug(f"ChitChatStackFrame processing: {user_message}")

        non_kb_message = self._resolve_non_kb_recommendation_message(tracker, user_message)
        if non_kb_message:
            logger.info("[EnterpriseSearchPolicy] 闲聊栈承接非知识库推荐上下文: %s", non_kb_message)
            return await self._handle_non_kb_recommendation(tracker, non_kb_message, pattern="chitchat")

        if self._should_route_chitchat_to_search(tracker, user_message):
            logger.info("[EnterpriseSearchPolicy] 补充条件被识别为闲聊，改走检索: %s", user_message)
            return await self._handle_search_frame(tracker, user_message)
        
        # 弹出栈帧
        tracker.dialogue_stack.pop()
        # 记录 Pattern 执行历史
        tracker.record_pattern("chitchat")
        
        if not user_message:
            return PolicyPrediction(
                action="action_send_text",
                confidence=0.8,
                policy_name=self.name,
                metadata={"text": "你好！有什么可以帮您的吗？"},
            )
        
        try:
            chitchat_answer = await self._generate_chitchat_answer(user_message)
            if chitchat_answer:
                chitchat_answer = self._sanitize_chitchat_answer(user_message, chitchat_answer)
                return PolicyPrediction(
                    action="action_send_text",
                    confidence=0.9,
                    policy_name=self.name,
                    metadata={
                        "text": chitchat_answer,
                        "degradation_reason": DegradationReason.CHITCHAT,
                    },
                )
        except Exception as e:
            logger.error(f"Chitchat generation error: {e}")
        
        # 默认回复
        return PolicyPrediction(
            action="action_send_text",
            confidence=0.7,
            policy_name=self.name,
            metadata={"text": "你好！很高兴和你聊天。"},
        )
    
    async def _handle_cannot_handle_frame(
        self,
        tracker: "DialogueStateTracker",
        frame: Any,
        domain: Optional["Domain"],
    ) -> PolicyPrediction:
        """处理CannotHandleStackFrame - 返回降级响应。"""
        logger.debug(f"CannotHandleStackFrame processing, reason: {getattr(frame, 'reason', '')}")
        
        # 弹出栈帧
        tracker.dialogue_stack.pop()
        # 记录 Pattern 执行历史
        tracker.record_pattern("cannot_handle")
        
        # 尝试从domain获取默认回复
        fallback_text = "抱歉，我没有理解您的意思。请换一种方式表达。"
        if domain:
            responses = domain.get_response("utter_default")
            if responses:
                import random
                fallback_text = random.choice(responses).text
        
        return PolicyPrediction(
            action="action_send_text",
            confidence=0.5,
            policy_name=self.name,
            metadata={
                "text": fallback_text,
                "degradation_reason": DegradationReason.CANNOT_HANDLE,
                "reason": getattr(frame, 'reason', ''),
            },
        )
    
    async def _handle_completed_frame(
        self,
        tracker: "DialogueStateTracker",
        frame: Any,
        domain: Optional["Domain"],
    ) -> PolicyPrediction:
        """处理CompletedStackFrame - 询问是否还有其他需求。
        
        当Flow完成后，系统会询问用户是否还有其他需要帮助的。
        """
        previous_flow = getattr(frame, 'previous_flow_name', '')
        logger.debug(f"CompletedStackFrame processing, previous_flow: {previous_flow}")
        
        # 弹出栈帧
        tracker.dialogue_stack.pop()
        # 记录 Pattern 执行历史
        tracker.record_pattern("completed")
        
        # 尝试从domain获取完成响应
        completed_text = "还有什么我可以帮您的吗？"
        if domain:
            responses = domain.get_response("utter_can_do_something_else")
            if responses:
                import random
                completed_text = random.choice(responses).text
        
        return PolicyPrediction(
            action="action_send_text",
            confidence=0.9,
            policy_name=self.name,
            metadata={
                "text": completed_text,
                "previous_flow": previous_flow,
            },
        )
    
    async def _handle_human_handoff_frame(
        self,
        tracker: "DialogueStateTracker",
        frame: Any,
        domain: Optional["Domain"],
    ) -> PolicyPrediction:
        """处理HumanHandoffStackFrame - 执行人工转接。
        
        当需要转接人工客服时，生成转接响应。
        """
        reason = getattr(frame, 'reason', '')
        logger.debug(f"HumanHandoffStackFrame processing, reason: {reason}")
        
        # 弹出栈帧
        tracker.dialogue_stack.pop()
        # 记录 Pattern 执行历史
        tracker.record_pattern("human_handoff")
        
        # 尝试从domain获取转接响应
        handoff_text = "好的，正在为您转接人工客服，请稍候..."
        if domain:
            responses = domain.get_response("utter_human_handoff")
            if responses:
                import random
                handoff_text = random.choice(responses).text
        
        return PolicyPrediction(
            action="action_send_text",
            confidence=0.95,
            policy_name=self.name,
            metadata={
                "text": handoff_text,
                "human_handoff": True,
                "reason": reason,
            },
        )
    
    async def _search(
        self,
        query: str,
        tracker: "DialogueStateTracker" = None,
    ) -> List[SearchResult]:
        """执行知识库搜索。
        
        Args:
            query: 搜索查询
            tracker: 对话状态追踪器（用于获取用户信息和历史对话）
            
        Returns:
            搜索结果列表
        """
        if not self._retriever:
            logger.debug("未配置检索器，跳过知识库搜索")
            return []
        
        try:
            logger.info(f"[EnterpriseSearchPolicy] 调用检索器: {type(self._retriever).__name__}")
            logger.info(f"[EnterpriseSearchPolicy] 查询: '{query}', top_k={self.config.retrieval.top_k}")
            
            # 构建 tracker_state 用于检索器获取用户信息和历史对话
            tracker_state = tracker.to_dict() if tracker else None
            
            # 调用检索器
            results = await self._retriever.search(
                query,
                top_k=self.config.retrieval.top_k,
                tracker_state=tracker_state,
            )
            
            logger.info(f"[EnterpriseSearchPolicy] 检索器返回 {len(results)} 条结果")
            
            # 过滤低相似度结果（score 为 None 时视为通过过滤）
            threshold = self.config.retrieval.similarity_threshold
            filtered = [
                r for r in results
                if r.score is None or r.score >= threshold
            ]
            
            logger.info(
                f"[EnterpriseSearchPolicy] 过滤后剩余 {len(filtered)} 条结果 "
                f"(阈值: {self.config.retrieval.similarity_threshold})"
            )
            
            return filtered
            
        except Exception as e:
            logger.error(f"[EnterpriseSearchPolicy] 搜索错误: {e}")
            return []
    
    async def _generate_rag_answer(
        self,
        question: str,
        search_results: List[SearchResult],
    ) -> Optional[str]:
        """生成RAG回答。
        
        Args:
            question: 用户问题
            search_results: 搜索结果
            
        Returns:
            生成的回答
        """
        if not search_results:
            return None
        
        # 构建上下文
        context_parts = []
        for i, result in enumerate(search_results, 1):
            source = result.source
            content = result.content
            # 格式：编号. 来源\n内容
            context_parts.append(f"{i}. {source}\n{content}")
        
        context = "\n\n".join(context_parts)
        logger.info(f"[EnterpriseSearchPolicy] RAG上下文:\n{context}")
        
        # 构建提示词
        prompt = self.RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )
        
        # 调用LLM
        try:
            response = await self.llm_client.complete([
                {"role": "user", "content": prompt}
            ])
            logger.info(f"[EnterpriseSearchPolicy] RAG回答: {response.content[:200] if response.content else 'None'}...")
            return response.content
        except Exception as e:
            logger.error(f"RAG generation error: {e}")
            return None
    
    async def _generate_chitchat_answer(self, message: str) -> Optional[str]:
        """生成闲聊回答。
        
        Args:
            message: 用户消息
            
        Returns:
            生成的回答
        """
        prompt = self.CHITCHAT_PROMPT_TEMPLATE.format(message=message)
        
        try:
            response = await self.llm_client.complete([
                {"role": "user", "content": prompt}
            ])
            return response.content
        except Exception as e:
            logger.error(f"Chitchat generation error: {e}")
            return None

    async def _generate_non_kb_recommendation_answer(self, message: str) -> Optional[str]:
        """知识库无命中时，基于通用知识生成推荐建议。"""
        prompt = self.NON_KB_RECOMMENDATION_PROMPT_TEMPLATE.format(message=message)

        try:
            response = await self.llm_client.complete([
                {"role": "user", "content": prompt}
            ])
            return response.content
        except Exception as e:
            logger.error(f"Non-KB recommendation generation error: {e}")
            return None
    
    def set_retriever(self, retriever: Any) -> None:
        """设置检索器。
        
        Args:
            retriever: 检索器实例
        """
        self._retriever = retriever


# 导出
__all__ = [
    "EnterpriseSearchPolicy",
    "EnterpriseSearchPolicyConfig",
    "SearchResult",  # 从base_retriever导入
]
