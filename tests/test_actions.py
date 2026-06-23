# -*- coding: utf-8 -*-

import unittest

from customer_agent.agent.actions import ActionSendText


class ActionSendTextTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_text_preserves_yunwen_public_metadata(self):
        action = ActionSendText()

        result = await action.run(
            tracker=None,
            text="answer",
            source="yunwen_mcp",
            citations=[{"title": "doc"}],
            image_urls=["http://example.com/a.jpg"],
            session_id="inspect-1",
            query_type="factual",
            crag_decision="accepted",
            search_results=["internal context should not be exposed"],
        )

        self.assertEqual(1, len(result.responses))
        self.assertEqual(
            {
                "text": "answer",
                "source": "yunwen_mcp",
                "citations": [{"title": "doc"}],
                "image_urls": ["http://example.com/a.jpg"],
                "session_id": "inspect-1",
                "query_type": "factual",
                "crag_decision": "accepted",
            },
            result.responses[0],
        )


if __name__ == "__main__":
    unittest.main()
