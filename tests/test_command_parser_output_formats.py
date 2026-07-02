# -*- coding: utf-8 -*-

import unittest

from customer_agent.dialogue_understanding.generator.command_parser import CommandParser


class CommandParserOutputFormatTests(unittest.TestCase):
    def setUp(self):
        self.parser = CommandParser()

    def test_parses_json_commands_output(self):
        result = self.parser.parse(
            '{"category":"flow_start","commands":["start flow cancel_order"],'
            '"dsl":"start flow cancel_order","intent":"cancel_order"}'
        )

        self.assertEqual([command.to_dsl() for command in result.commands], [
            "start flow cancel_order",
        ])

    def test_parses_json_dsl_output(self):
        result = self.parser.parse(
            '{"category":"slot_fill","dsl":"set slot order_id \\"1001003\\""}'
        )

        self.assertEqual([command.to_dsl() for command in result.commands], [
            "set slot order_id 1001003",
        ])

    def test_ignores_qwen_thinking_before_json_output(self):
        result = self.parser.parse(
            '<think>先判断用户意图。</think>\n'
            '{"commands":["set slot if_cancel_order false"]}'
        )

        self.assertEqual([command.to_dsl() for command in result.commands], [
            "set slot if_cancel_order False",
        ])

    def test_recovers_json_output_missing_opening_brace(self):
        result = self.parser.parse(
            '"category":"boundary","commands":["human_handoff"],'
            '"dsl":"human_handoff","intent":"human_handoff"}'
        )

        self.assertEqual([command.to_dsl() for command in result.commands], [
            "human_handoff",
        ])


if __name__ == "__main__":
    unittest.main()
