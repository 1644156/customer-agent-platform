# -*- coding: utf-8 -*-
"""Smoke test the fine-tuned command parser endpoint.

This checks only the command-parser LLM path:
endpoints.yml -> OpenAI-compatible EAS/vLLM service -> CommandParser.
It does not execute flows or touch MySQL/Neo4j.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from customer_agent.dialogue_understanding.generator.command_parser import CommandParser
from customer_agent.shared.config import EndpointsConfig
from customer_agent.shared.llm import create_llm_client


DEFAULT_APP_DIR = Path("commerce_service_app")
SYSTEM_PROMPT = (
    "你是电商客服 Agent 的命令解析器。只根据用户最新消息和当前状态输出可执行命令，"
    "不要直接回答用户。启动业务流程时只输出 start flow；正在收集槽位时只设置当前槽位。"
)


def build_messages(message: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                '当前状态: {"active_flow": null, "requested_slot": null, "known_slots": {}}\n'
                f"用户消息: {message}"
            ),
        },
    ]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", type=Path, default=DEFAULT_APP_DIR)
    parser.add_argument("--model", default="command_parser_ft")
    parser.add_argument("--message", default="我要取消订单")
    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    dotenv.load_dotenv(args.app_dir / ".env", override=True)

    endpoints_path = args.app_dir / "endpoints.yml"
    model_config = EndpointsConfig.load(endpoints_path).get_model_config(args.model)
    if model_config is None:
        raise SystemExit(f"Model config not found: {args.model}")

    client = create_llm_client(
        type=model_config.type,
        model=model_config.model,
        api_key=model_config.api_key,
        api_base=model_config.api_base,
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
        timeout=model_config.timeout,
        enable_thinking=model_config.enable_thinking,
        extra_body=model_config.extra_body,
    )

    response = await client.complete(build_messages(args.message))
    parse_result = CommandParser().parse(response.content)
    commands = [command.to_dsl() for command in parse_result.commands]

    print(f"model: {response.model}")
    print(f"raw: {response.content}")
    print(f"commands: {commands}")

    if not commands:
        raise SystemExit("No commands parsed from fine-tuned model output.")


if __name__ == "__main__":
    asyncio.run(main())
