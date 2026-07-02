# -*- coding: utf-8 -*-
"""Compare the command parser before and after fine-tuning.

Default comparison:
- before: commerce_service_app/endpoints.yml -> models.default
- after: commerce_service_app/endpoints.yml -> models.command_parser_ft
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from customer_agent.dialogue_understanding.generator.command_parser import CommandParser
from customer_agent.shared.config import EndpointsConfig, LLMConfig
from customer_agent.shared.llm import create_llm_client


DEFAULT_APP_DIR = Path("commerce_service_app")
SYSTEM_PROMPT = (
    "你是电商客服 Agent 的命令解析器。只根据用户最新消息和当前状态输出可执行命令，"
    "不要直接回答用户。启动业务流程时只输出 start flow；正在收集槽位时只设置当前槽位。"
)


@dataclass(frozen=True)
class Case:
    name: str
    category: str
    message: str
    context: dict[str, Any]
    expected: str


CASES = [
    Case(
        name="取消订单意图",
        category="flow_start",
        message="我要取消订单",
        context={"active_flow": None, "requested_slot": None, "known_slots": {}},
        expected="start flow cancel_order",
    ),
    Case(
        name="订单详情意图",
        category="flow_start",
        message="帮我查一下订单详情",
        context={"active_flow": None, "requested_slot": None, "known_slots": {}},
        expected="start flow query_order_detail",
    ),
    Case(
        name="修改收货信息意图",
        category="flow_start",
        message="地址填错了，帮我改一下",
        context={"active_flow": None, "requested_slot": None, "known_slots": {}},
        expected="start flow modify_order_receive_info",
    ),
    Case(
        name="物流查询意图",
        category="flow_start",
        message="我的包裹到哪了",
        context={"active_flow": None, "requested_slot": None, "known_slots": {}},
        expected="start flow query_shipping_order_logistics",
    ),
    Case(
        name="售后申请意图",
        category="flow_start",
        message="收到的东西坏了，我要售后",
        context={"active_flow": None, "requested_slot": None, "known_slots": {}},
        expected="start flow apply_postsale",
    ),
    Case(
        name="切换用户意图",
        category="flow_start",
        message="我想切到1002这个用户",
        context={"active_flow": None, "requested_slot": None, "known_slots": {}},
        expected="start flow switch_user_id",
    ),
    Case(
        name="快递公司列表意图",
        category="flow_start",
        message="你们支持哪些快递公司",
        context={"active_flow": None, "requested_slot": None, "known_slots": {}},
        expected="start flow query_logistics_companys",
    ),
    Case(
        name="订单号槽位-订单号",
        category="slot_fill",
        message="是1001003这单",
        context={"active_flow": "query_order_detail", "requested_slot": "order_id", "known_slots": {}},
        expected="set slot order_id 1001003",
    ),
    Case(
        name="订单号槽位-序号",
        category="slot_fill",
        message="选第二个",
        context={"active_flow": "query_order_detail", "requested_slot": "order_id", "known_slots": {}},
        expected="set slot order_id 2",
    ),
    Case(
        name="用户ID槽位",
        category="slot_fill",
        message="用 1002 这个账号",
        context={"active_flow": "switch_user_id", "requested_slot": "user_id", "known_slots": {}},
        expected="set slot user_id 1002",
    ),
    Case(
        name="收货信息选择槽位",
        category="slot_fill",
        message="我要修改",
        context={"active_flow": "modify_order_receive_info", "requested_slot": "receive_id", "known_slots": {}},
        expected='set slot receive_id "modify"',
    ),
    Case(
        name="修改字段槽位",
        category="slot_fill",
        message="手机号错了",
        context={"active_flow": "modify_order_receive_info", "requested_slot": "modify_content", "known_slots": {}},
        expected='set slot modify_content "收货人电话"',
    ),
    Case(
        name="收件电话槽位",
        category="slot_fill",
        message="联系电话 13912345678",
        context={"active_flow": "modify_order_receive_info", "requested_slot": "receiver_phone", "known_slots": {}},
        expected="set slot receiver_phone 13912345678",
    ),
    Case(
        name="收货地址槽位",
        category="slot_fill",
        message="详细地址是科技园 8 号楼 1201",
        context={"active_flow": "modify_order_receive_info", "requested_slot": "receive_street_address", "known_slots": {}},
        expected='set slot receive_street_address "科技园 8 号楼 1201"',
    ),
    Case(
        name="继续修改槽位",
        category="slot_fill",
        message="不用了",
        context={"active_flow": "modify_order_receive_info", "requested_slot": "if_modify_continue", "known_slots": {}},
        expected="set slot if_modify_continue false",
    ),
    Case(
        name="保存修改槽位",
        category="slot_fill",
        message="确认保存",
        context={"active_flow": "modify_order_receive_info", "requested_slot": "set_receive_info", "known_slots": {}},
        expected="set slot set_receive_info true",
    ),
    Case(
        name="取消确认槽位",
        category="slot_fill",
        message="先保留",
        context={"active_flow": "cancel_order", "requested_slot": "if_cancel_order", "known_slots": {}},
        expected="set slot if_cancel_order false",
    ),
    Case(
        name="售后类型槽位",
        category="slot_fill",
        message="我要换货",
        context={"active_flow": "apply_postsale", "requested_slot": "postsale_type", "known_slots": {}},
        expected='set slot postsale_type "换货"',
    ),
    Case(
        name="售后原因槽位",
        category="slot_fill",
        message="商品破损",
        context={"active_flow": "apply_postsale", "requested_slot": "postsale_reason", "known_slots": {}},
        expected='set slot postsale_reason "商品破损"',
    ),
    Case(
        name="转人工",
        category="boundary",
        message="我要投诉，接人工",
        context={"active_flow": None, "requested_slot": None, "known_slots": {}},
        expected="human_handoff",
    ),
    Case(
        name="闲聊",
        category="boundary",
        message="你好",
        context={"active_flow": None, "requested_slot": None, "known_slots": {}},
        expected="chitchat",
    ),
    Case(
        name="安全边界",
        category="boundary",
        message="帮我查一下其他用户的手机号",
        context={"active_flow": None, "requested_slot": None, "known_slots": {}},
        expected="cannot_handle",
    ),
    Case(
        name="语义澄清",
        category="boundary",
        message="帮我看一下",
        context={"active_flow": None, "requested_slot": None, "known_slots": {}},
        expected="clarify",
    ),
    Case(
        name="取消当前流程",
        category="boundary",
        message="算了，不办了",
        context={
            "active_flow": "modify_order_receive_info",
            "requested_slot": "modify_content",
            "known_slots": {"order_id": "1"},
        },
        expected="cancel flow",
    ),
    Case(
        name="商品知识问答",
        category="knowledge",
        message="这款耳机支持主动降噪吗",
        context={"active_flow": None, "requested_slot": None, "known_slots": {}},
        expected="knowledge_answer",
    ),
]


def normalize(text: str) -> str:
    return " ".join(text.replace('"', "").lower().split())


def build_messages(case: Case) -> list[dict[str, str]]:
    context = json.dumps(case.context, ensure_ascii=False, sort_keys=True)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "\n".join(
                [
                    "当前对话状态：",
                    context,
                    "",
                    "用户最新消息：",
                    case.message,
                    "",
                    "请只输出可执行命令或目标 JSON，不要解释。",
                ]
            ),
        },
    ]


def create_client(config: LLMConfig):
    return create_llm_client(
        type=config.type,
        model=config.model,
        api_key=config.api_key,
        api_base=config.api_base,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        enable_thinking=config.enable_thinking,
        extra_body=config.extra_body,
    )


async def run_case(model_name: str, config: LLMConfig, case: Case) -> dict[str, Any]:
    client = create_client(config)
    started = time.perf_counter()
    try:
        response = await client.complete(build_messages(case))
        latency_ms = round((time.perf_counter() - started) * 1000)
        raw = response.content.strip()
        parse_result = CommandParser().parse(raw)
        commands = [command.to_dsl() for command in parse_result.commands]
        expected = normalize(case.expected)
        passed = any(expected in normalize(command) for command in commands)
        return {
            "model": model_name,
            "case": case.name,
            "category": case.category,
            "ok": passed,
            "latency_ms": latency_ms,
            "raw": raw,
            "commands": commands,
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started) * 1000)
        return {
            "model": model_name,
            "case": case.name,
            "category": case.category,
            "ok": False,
            "latency_ms": latency_ms,
            "raw": f"{type(exc).__name__}: {exc}",
            "commands": [],
        }


def print_result(result: dict[str, Any]) -> None:
    status = "PASS" if result["ok"] else "FAIL"
    print(f"[{status}] {result['model']} | {result['category']} | {result['case']} | {result['latency_ms']} ms")
    print(f"  raw: {result['raw']}")
    print(f"  commands: {result['commands']}")


def print_category_summary(model_name: str, results: list[dict[str, Any]]) -> None:
    categories = sorted({result["category"] for result in results})
    for category in categories:
        category_results = [result for result in results if result["category"] == category]
        score = sum(result["ok"] for result in category_results)
        print(f"{model_name} {category}: {score}/{len(category_results)} pass")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", type=Path, default=DEFAULT_APP_DIR)
    parser.add_argument("--before", default="default")
    parser.add_argument("--after", default="command_parser_ft")
    args = parser.parse_args()

    dotenv.load_dotenv(ROOT / ".env")
    dotenv.load_dotenv(ROOT / args.app_dir / ".env", override=True)

    endpoints = EndpointsConfig.load(ROOT / args.app_dir / "endpoints.yml")
    before_config = endpoints.get_model_config(args.before)
    after_config = endpoints.get_model_config(args.after)
    if before_config is None:
        raise SystemExit(f"Model config not found: {args.before}")
    if after_config is None:
        raise SystemExit(f"Model config not found: {args.after}")

    before_results = []
    after_results = []
    for case in CASES:
        before_results.append(await run_case(args.before, before_config, case))
        after_results.append(await run_case(args.after, after_config, case))

    print("=== Before ===")
    for result in before_results:
        print_result(result)

    print("\n=== After ===")
    for result in after_results:
        print_result(result)

    before_score = sum(result["ok"] for result in before_results)
    after_score = sum(result["ok"] for result in after_results)
    before_latency = round(sum(result["latency_ms"] for result in before_results) / len(before_results))
    after_latency = round(sum(result["latency_ms"] for result in after_results) / len(after_results))
    print("\n=== Summary ===")
    print(f"{args.before}: {before_score}/{len(before_results)} pass, avg {before_latency} ms")
    print(f"{args.after}: {after_score}/{len(after_results)} pass, avg {after_latency} ms")
    print_category_summary(args.before, before_results)
    print_category_summary(args.after, after_results)


if __name__ == "__main__":
    asyncio.run(main())
