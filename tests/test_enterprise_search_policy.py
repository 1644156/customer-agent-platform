# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace

from customer_agent.policies.enterprise_search_policy import (
    EnterpriseSearchPolicy,
    EnterpriseSearchPolicyConfig,
)
from customer_agent.retrieval.base_retriever import SearchResult
from customer_agent.shared.constants import DegradationReason


class FakeStack:
    def __init__(self):
        self.popped = 0

    def pop(self):
        self.popped += 1


class FakeTracker:
    def __init__(self):
        self.dialogue_stack = FakeStack()
        self.patterns = []
        self.latest_message = SimpleNamespace(text="推荐一下手机，拍照用的")
        self.messages = [{"role": "user", "content": self.latest_message.text}]
        self.slots = {"user_id": "35"}

    def record_pattern(self, pattern):
        self.patterns.append(pattern)

    def to_dict(self):
        return {"slots": {"user_id": {"value": "35"}}, "events": []}

    def get_messages_for_llm(self, max_turns=10):
        return self.messages[-max_turns:]

    def get_slot(self, name):
        return self.slots.get(name)

    def set_slot(self, name, value):
        self.slots[name] = value


class KeyboardFollowupTracker(FakeTracker):
    def __init__(self):
        super().__init__()
        self.latest_message = SimpleNamespace(text="用来打游戏，预算1000内吧")
        self.messages = [
            {"role": "user", "content": "推荐一下键盘"},
            {
                "role": "assistant",
                "content": "方便问下你主要用来打字、游戏，还是有其他需求？另外预算大概多少？",
            },
            {"role": "user", "content": self.latest_message.text},
        ]


class EmptyRetriever:
    async def search(self, query, top_k=3, tracker_state=None):
        return []


class SequenceRetriever:
    def __init__(self, responses):
        self.responses = list(responses)
        self.queries = []

    async def search(self, query, top_k=3, tracker_state=None):
        self.queries.append(query)
        if self.responses:
            return self.responses.pop(0)
        return []


class HitRetriever:
    async def search(self, query, top_k=3, tracker_state=None):
        return [SearchResult(text="商品名称: OPPO Find X8手机 AI拍照新机", score=1.0)]


class YunwenMcpHitRetriever:
    async def search(self, query, top_k=3, tracker_state=None):
        return [
            SearchResult(
                text="Yunwen final answer about Huawei Qingyun B730. Keep this exact response.",
                score=1.0,
                metadata={
                    "source": "yunwen_mcp",
                    "session_id": "inspect-1",
                    "citations": [{"title": "Huawei Qingyun B730"}],
                    "image_urls": ["http://example.com/b730.jpg"],
                    "query_type": "factual",
                    "crag_decision": "accepted",
                },
            )
        ]


class RecordingRetriever:
    def __init__(self):
        self.queries = []

    async def search(self, query, top_k=3, tracker_state=None):
        self.queries.append(query)
        return [SearchResult(text="商品名称: 京东京造 JZ990 机械键盘 游戏键盘 RGB 红轴", score=1.0)]


class FakeLLM:
    def __init__(self, response="[NO_RAG_ANSWER]"):
        self.calls = []
        self.response = response

    async def complete(self, messages):
        self.calls.append(messages[0]["content"])
        return SimpleNamespace(content=self.response)


class QueueLLM:
    def __init__(self, responses):
        self.calls = []
        self.responses = list(responses)

    async def complete(self, messages):
        self.calls.append(messages[0]["content"])
        return SimpleNamespace(content=self.responses.pop(0))


class EnterpriseSearchPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_search_frame_falls_back_to_chitchat_when_enabled(self):
        config = EnterpriseSearchPolicyConfig()
        config.chitchat_enabled = True
        llm = FakeLLM(response="可以的，我先按热门和通用口味给你推荐。")
        policy = EnterpriseSearchPolicy(
            config=config,
            llm_client=llm,
            retriever=EmptyRetriever(),
        )
        tracker = FakeTracker()

        prediction = await policy._handle_search_frame(
            tracker,
            "推荐一下手机，拍照用的",
        )

        self.assertEqual("action_send_text", prediction.action)
        self.assertIn("热门", prediction.metadata["text"])
        self.assertEqual(
            DegradationReason.CHITCHAT,
            prediction.metadata["degradation_reason"],
        )
        self.assertEqual(1, len(llm.calls))
        self.assertIn("不要编造具体SKU", llm.calls[0])
        self.assertEqual(1, tracker.dialogue_stack.popped)

    async def test_empty_search_chitchat_does_not_return_ungrounded_product_names(self):
        config = EnterpriseSearchPolicyConfig()
        config.chitchat_enabled = True
        llm = FakeLLM(response="可以考虑 iPhone 15 Pro，拍照和性能都很强。")
        policy = EnterpriseSearchPolicy(
            config=config,
            llm_client=llm,
            retriever=EmptyRetriever(),
        )
        tracker = FakeTracker()

        prediction = await policy._handle_search_frame(
            tracker,
            "推荐一下手机，拍照用的",
        )

        self.assertEqual("action_send_text", prediction.action)
        self.assertNotIn("iPhone 15 Pro", prediction.metadata["text"])
        self.assertIn("用途、预算和关键参数", prediction.metadata["text"])
        self.assertEqual(
            DegradationReason.CHITCHAT,
            prediction.metadata["degradation_reason"],
        )

    async def test_no_rag_answer_returns_grounded_search_result(self):
        config = EnterpriseSearchPolicyConfig()
        config.chitchat_enabled = True
        llm = FakeLLM()
        policy = EnterpriseSearchPolicy(
            config=config,
            llm_client=llm,
            retriever=HitRetriever(),
        )
        tracker = FakeTracker()

        prediction = await policy._handle_search_frame(
            tracker,
            "推荐一下手机，拍照用的",
        )

        self.assertEqual("action_send_text", prediction.action)
        self.assertIn("OPPO Find X8", prediction.metadata["text"])
        self.assertEqual(1, len(llm.calls))
        self.assertEqual(1, tracker.dialogue_stack.popped)

    async def test_yunwen_mcp_result_is_sent_directly_without_second_rag(self):
        config = EnterpriseSearchPolicyConfig()
        config.chitchat_enabled = True
        llm = FakeLLM(response="This local RAG answer should not be used.")
        policy = EnterpriseSearchPolicy(
            config=config,
            llm_client=llm,
            retriever=YunwenMcpHitRetriever(),
        )
        tracker = FakeTracker()

        prediction = await policy._handle_search_frame(
            tracker,
            "Tell me about Huawei Qingyun B730",
        )

        self.assertEqual("action_send_text", prediction.action)
        self.assertEqual(
            "Yunwen final answer about Huawei Qingyun B730. Keep this exact response.",
            prediction.metadata["text"],
        )
        self.assertEqual([], llm.calls)
        self.assertEqual("yunwen_mcp", prediction.metadata["source"])
        self.assertEqual([{"title": "Huawei Qingyun B730"}], prediction.metadata["citations"])
        self.assertEqual(["http://example.com/b730.jpg"], prediction.metadata["image_urls"])
        self.assertEqual("inspect-1", prediction.metadata["session_id"])
        self.assertEqual(1, tracker.dialogue_stack.popped)

    async def test_followup_preference_uses_previous_recommendation_context(self):
        config = EnterpriseSearchPolicyConfig()
        config.chitchat_enabled = True
        llm = FakeLLM(response="可以优先看京东京造 JZ990 机械键盘。")
        retriever = RecordingRetriever()
        policy = EnterpriseSearchPolicy(
            config=config,
            llm_client=llm,
            retriever=retriever,
        )
        tracker = KeyboardFollowupTracker()

        prediction = await policy._handle_chitchat_frame(
            tracker,
            "用来打游戏，预算1000内吧",
        )

        self.assertEqual("action_send_text", prediction.action)
        self.assertEqual(1, len(retriever.queries))
        self.assertIn("推荐一下键盘", retriever.queries[0])
        self.assertIn("预算1000内", retriever.queries[0])
        self.assertIn("机械键盘", prediction.metadata["text"])
        self.assertEqual(1, tracker.dialogue_stack.popped)

    async def test_empty_recommendation_followup_uses_non_kb_generation_without_retrieval(self):
        config = EnterpriseSearchPolicyConfig()
        config.chitchat_enabled = True
        llm = QueueLLM([
            "台灯可以先看学习护眼、氛围照明或工作台使用。",
            "学习护眼台灯建议重点看照度、频闪控制、色温调节和显色指数。",
        ])
        retriever = SequenceRetriever([
            [],
            [SearchResult(text="商品名称: Apple iPhone 12 手机", score=1.0)],
        ])
        policy = EnterpriseSearchPolicy(
            config=config,
            llm_client=llm,
            retriever=retriever,
        )
        tracker = FakeTracker()
        tracker.latest_message = SimpleNamespace(text="推荐一下台灯")
        tracker.messages = [{"role": "user", "content": "推荐一下台灯"}]

        first = await policy._handle_search_frame(tracker, "推荐一下台灯")
        self.assertEqual("action_send_text", first.action)
        self.assertEqual(["推荐一下台灯"], retriever.queries)
        self.assertIsNotNone(
            tracker.get_slot(EnterpriseSearchPolicy.NON_KB_RECOMMENDATION_SLOT)
        )

        tracker.latest_message = SimpleNamespace(text="学习护眼")
        tracker.messages.append({"role": "assistant", "content": first.metadata["text"]})
        tracker.messages.append({"role": "user", "content": "学习护眼"})

        second = await policy._handle_chitchat_frame(tracker, "学习护眼")

        self.assertEqual("action_send_text", second.action)
        self.assertEqual(["推荐一下台灯"], retriever.queries)
        self.assertIn("学习护眼台灯", second.metadata["text"])
        self.assertTrue(second.metadata["non_kb_recommendation"])
        self.assertIn("推荐一下台灯；补充条件：学习护眼", llm.calls[-1])

    async def test_empty_recommendation_followup_search_frame_also_skips_retrieval(self):
        config = EnterpriseSearchPolicyConfig()
        config.chitchat_enabled = True
        llm = QueueLLM([
            "台灯可以先看学习护眼、氛围照明或工作台使用。",
            "学习护眼台灯建议重点看照度、频闪控制、色温调节和显色指数。",
        ])
        retriever = SequenceRetriever([
            [],
            [SearchResult(text="商品名称: Apple iPhone 12 手机", score=1.0)],
        ])
        policy = EnterpriseSearchPolicy(
            config=config,
            llm_client=llm,
            retriever=retriever,
        )
        tracker = FakeTracker()

        first = await policy._handle_search_frame(tracker, "推荐一下台灯")
        self.assertEqual("action_send_text", first.action)

        second = await policy._handle_search_frame(tracker, "学习护眼")

        self.assertEqual("action_send_text", second.action)
        self.assertEqual(["推荐一下台灯"], retriever.queries)
        self.assertIn("学习护眼台灯", second.metadata["text"])
        self.assertTrue(second.metadata["non_kb_recommendation"])

    async def test_non_kb_context_does_not_capture_unrelated_short_chitchat(self):
        config = EnterpriseSearchPolicyConfig()
        config.chitchat_enabled = True
        llm = QueueLLM([
            "台灯可以先看学习护眼、氛围照明或工作台使用。",
            "我是一个客服助手，可以帮你处理咨询和推荐问题。",
        ])
        retriever = SequenceRetriever([[]])
        policy = EnterpriseSearchPolicy(
            config=config,
            llm_client=llm,
            retriever=retriever,
        )
        tracker = FakeTracker()

        await policy._handle_search_frame(tracker, "推荐一下台灯")
        second = await policy._handle_chitchat_frame(tracker, "你是谁")

        self.assertEqual("action_send_text", second.action)
        self.assertEqual(["推荐一下台灯"], retriever.queries)
        self.assertFalse(second.metadata.get("non_kb_recommendation"))
        self.assertNotIn("补充条件", llm.calls[-1])


if __name__ == "__main__":
    unittest.main()
