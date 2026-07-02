# -*- coding: utf-8 -*-
"""Generate a synthetic command-parser SFT dataset for the customer agent.

The dataset trains a small model to convert the latest user message plus a
compact dialogue state into this project's command DSL. It intentionally does
not train the model to answer customers directly; business execution stays in
Flow / Policy / Action code.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TASK = "customer_agent_command_parser"
SOURCE = "synthetic_v1"
SYSTEM_PROMPT = (
    "你是电商客服 Agent 的命令解析器。只根据用户最新消息和当前状态输出可执行命令，"
    "不要直接回答用户。启动业务流程时只输出 start flow；正在收集槽位时只设置当前槽位。"
)


FLOW_UTTERANCES: dict[str, list[str]] = {
    "switch_user_id": [
        "帮我切换一下账号",
        "我想换个用户身份",
        "切到另一个 user_id",
        "我要用别的账号查订单",
        "把当前用户改一下",
        "换成另一个用户",
        "切换用户",
        "现在不是我的账号，帮我换一下",
        "我想指定新的用户ID",
        "把会话用户换掉",
        "我需要切换到测试用户",
        "帮我改一下当前登录用户",
        "用户身份不对，换一个",
        "先切账号再查",
        "我想换成 1002 这个用户",
        "可以切换 customer id 吗",
        "帮我切到别的用户数据",
        "换一个演示用户",
    ],
    "query_order_detail": [
        "帮我查一下订单详情",
        "我想看看订单信息",
        "查一下我最近的订单",
        "订单现在是什么状态",
        "帮我看看这个订单买了什么",
        "我想查询订单明细",
        "给我看下订单详情",
        "查订单",
        "我刚下的单在哪里看",
        "这个订单的收货信息和商品是什么",
        "帮我确认一下订单内容",
        "我需要查询订单状态",
        "看一下订单有没有完成",
        "订单详情帮我拉一下",
        "我想知道订单里有哪些商品",
        "帮我看订单号对应的详情",
        "查一下订单是否付款成功",
        "看下三天内完成的订单",
    ],
    "modify_order_receive_info": [
        "我要修改订单收货信息",
        "帮我改一下收货地址",
        "订单地址填错了",
        "收货人电话需要改",
        "收件人姓名写错了",
        "我想换个收货地址",
        "订单还没到，能改地址吗",
        "帮我修改这单的收货人",
        "收货信息不对，需要调整",
        "把订单的联系电话改一下",
        "我想更新订单配送地址",
        "地址写成老家的了，帮我改",
        "这个订单能不能改收货信息",
        "需要修改订单上的手机号",
        "帮我把收货地址改成公司",
        "订单收货资料想重新填",
        "我想改一下配送信息",
        "收货地址有误",
    ],
    "cancel_order": [
        "我要取消订单",
        "这单不想要了",
        "帮我退掉这个未发货订单",
        "我想取消刚才下的单",
        "这个订单能撤销吗",
        "不买了，帮我取消",
        "订单还没发货的话取消一下",
        "帮我关闭订单",
        "这笔订单不要了",
        "我要取消购买",
        "这单下错了",
        "取消一下我的订单",
        "能不能帮我取消待付款订单",
        "订单买错型号了，取消",
        "先别发货，帮我取消订单",
        "撤掉这单",
        "我想作废订单",
        "不需要了，取消订单",
    ],
    "query_logistics_companys": [
        "你们支持哪些快递公司",
        "能查哪些物流承运商",
        "支持的快递列表给我看下",
        "有哪些物流公司可以选",
        "帮我看看支持什么快递",
        "可以走顺丰吗",
        "平台支持哪些配送公司",
        "查一下快递公司列表",
        "你们能发哪些物流",
        "支持的物流商有哪些",
        "承运商列表",
        "能不能查支持的快递",
        "有哪些合作快递",
        "我想知道可用物流公司",
        "发货可以选择哪些快递",
        "物流公司有哪些",
        "支持京东物流吗",
        "配送服务商列表",
    ],
    "query_shipping_order_logistics": [
        "查一下订单物流",
        "我的包裹到哪了",
        "帮我看看物流信息",
        "订单发货了吗",
        "快递现在在哪里",
        "查询运输进度",
        "这个订单的快递轨迹给我看下",
        "物流怎么还没更新",
        "帮我查发货后的物流",
        "看下包裹配送状态",
        "我想知道快递单进度",
        "订单运输情况怎么样",
        "查一下已经发货订单的物流",
        "物流详情",
        "包裹预计什么时候到",
        "我的订单配送到哪一步了",
        "查快递",
        "看看这单的物流轨迹",
    ],
    "apply_postsale": [
        "我要申请售后",
        "我想退货",
        "帮我申请换货",
        "商品有问题，需要售后",
        "收到的东西坏了",
        "我要退货退款",
        "这个商品不合适，想退",
        "可以申请售后吗",
        "帮我处理一下退换货",
        "我想提交售后申请",
        "商品少发了，需要售后",
        "东西破损了，帮我换一个",
        "我要发起售后",
        "买错了想退货",
        "质量有问题，申请退货",
        "收到后发现不能用",
        "帮我走一下售后流程",
        "申请售后服务",
    ],
}


SLOT_CASES: list[dict[str, Any]] = [
    {
        "intent": "switch_user_id",
        "active_flow": "switch_user_id",
        "slot": "user_id",
        "values": [
            ("1001", ["1001", "切到1001", "用户 ID 是 1001", "换成用户1001"]),
            ("1002", ["1002", "用 1002 这个账号", "切换到1002", "当前用户改成1002"]),
            ("2001", ["2001", "测试用户 2001", "换到2001", "账号是2001"]),
        ],
    },
    {
        "intent": "query_order_detail",
        "active_flow": "query_order_detail",
        "slot": "order_id",
        "values": [
            ("1", ["选第一个", "第一个订单", "1", "就第1个"]),
            ("2", ["第二个", "选 2", "我要看第2单", "2"]),
            ("1001003", ["订单号1001003", "查 1001003", "是1001003这单", "订单 ID 1001003"]),
        ],
    },
    {
        "intent": "modify_order_receive_info",
        "active_flow": "modify_order_receive_info",
        "slot": "receive_id",
        "values": [
            ("1", ["选第一个地址", "1", "用第一个收货信息", "就默认第一个"]),
            ("modify", ["我要修改", "新建一个地址", "改成新的收货信息", "选择修改"]),
            (False, ["不改了", "算了", "取消修改", "先不处理"]),
        ],
    },
    {
        "intent": "modify_order_receive_info",
        "active_flow": "modify_order_receive_info",
        "slot": "modify_content",
        "values": [
            ("收货人姓名", ["改姓名", "收货人名字错了", "修改收件人姓名", "把姓名改一下"]),
            ("收货人电话", ["改电话", "手机号错了", "修改收货人电话", "换个联系电话"]),
            ("收货地址", ["改地址", "地址错了", "修改收货地址", "换配送地址"]),
            ("取消", ["不修改了", "取消", "算了不改", "先不改了"]),
        ],
    },
    {
        "intent": "modify_order_receive_info",
        "active_flow": "modify_order_receive_info",
        "slot": "receiver_name",
        "values": [
            ("张三", ["张三", "收货人写张三", "改成张三", "姓名是张三"]),
            ("李雷", ["李雷", "收件人李雷", "名字改李雷", "写李雷"]),
            ("王小明", ["王小明", "收货人改为王小明", "名字是王小明", "王小明收"]),
        ],
    },
    {
        "intent": "modify_order_receive_info",
        "active_flow": "modify_order_receive_info",
        "slot": "receiver_phone",
        "values": [
            ("13800138000", ["13800138000", "电话改成13800138000", "手机号是 13800138000", "收件电话13800138000"]),
            ("13912345678", ["13912345678", "改为13912345678", "联系电话 13912345678", "手机号13912345678"]),
            ("13688889999", ["13688889999", "写13688889999", "电话是13688889999", "收货电话改13688889999"]),
        ],
    },
    {
        "intent": "modify_order_receive_info",
        "active_flow": "modify_order_receive_info",
        "slot": "receive_street_address",
        "values": [
            ("科技园 8 号楼 1201", ["科技园8号楼1201", "详细地址是科技园 8 号楼 1201", "街道地址填科技园8号楼1201", "写科技园 8 号楼 1201"]),
            ("人民路 66 号 3 单元 502", ["人民路66号3单元502", "地址写人民路 66 号 3 单元 502", "详细地址人民路66号3单元502", "填人民路66号3单元502"]),
        ],
    },
    {
        "intent": "modify_order_receive_info",
        "active_flow": "modify_order_receive_info",
        "slot": "if_modify_continue",
        "values": [
            (True, ["继续改", "还要修改", "是的继续", "还有别的信息要改"]),
            (False, ["不继续了", "不用了", "否", "就这些"]),
        ],
    },
    {
        "intent": "modify_order_receive_info",
        "active_flow": "modify_order_receive_info",
        "slot": "set_receive_info",
        "values": [
            (True, ["确认修改", "是的，提交", "确认保存", "没问题就按这个改"]),
            (False, ["不确认", "先不要保存", "取消提交", "算了别改"]),
        ],
    },
    {
        "intent": "cancel_order",
        "active_flow": "cancel_order",
        "slot": "if_cancel_order",
        "values": [
            (True, ["确认取消", "是的取消", "对，帮我取消", "确定不要了"]),
            (False, ["不取消了", "先保留", "算了不要取消", "否"]),
        ],
    },
    {
        "intent": "apply_postsale",
        "active_flow": "apply_postsale",
        "slot": "postsale_type",
        "values": [
            ("退货", ["我要退货", "选择退货", "退货退款", "申请退货"]),
            ("换货", ["我要换货", "换一个新的", "选择换货", "申请换货"]),
        ],
    },
    {
        "intent": "apply_postsale",
        "active_flow": "apply_postsale",
        "slot": "postsale_reason",
        "values": [
            ("商品破损", ["商品破损", "收到的时候外壳碎了", "东西坏了", "包装破损影响使用"]),
            ("尺码不合适", ["尺码不合适", "穿不上", "尺寸买小了", "大小不合适"]),
            ("质量问题", ["质量有问题", "用了一次就坏了", "产品不能正常使用", "做工有瑕疵"]),
        ],
    },
]


KNOWLEDGE_UTTERANCES = [
    "这款耳机支持主动降噪吗",
    "有没有适合学生党的手机推荐",
    "帮我推荐一款适合办公的笔记本",
    "这个空气炸锅能不能烤红薯",
    "云问文档里说怎么导入 PDF",
    "产品说明书里面保修期是多久",
    "这款路由器支持 mesh 组网吗",
    "有没有适合送父母的按摩仪",
    "我想了解这个商品和上一代的区别",
    "文档里提到的安装步骤是什么",
    "这台洗地机续航怎么样",
    "帮我查一下说明书里怎么清洗滤网",
    "有没有 500 元以内的蓝牙音箱",
    "商品参数里有没有防水等级",
    "这款手机适合拍视频吗",
    "文档问答：如何配置 MCP 服务",
    "售后政策里写的退货条件是什么",
    "这个型号支持哪些配件",
    "推荐一款适合跑步的耳机",
    "知识库里有没有部署说明",
    "商品推荐一下",
    "这件商品有什么卖点",
]

CHITCHAT_UTTERANCES = [
    "你好",
    "早上好",
    "谢谢你",
    "你是谁",
    "今天天气怎么样",
    "哈哈还挺聪明",
    "辛苦了",
    "你能陪我聊会儿吗",
    "晚安",
    "你叫什么名字",
    "不错不错",
    "我先看看",
    "没事了",
    "好的谢谢",
    "你真快",
]

HUMAN_HANDOFF_UTTERANCES = [
    "转人工",
    "我要找人工客服",
    "让真人来处理",
    "这个问题你解决不了，帮我转人工",
    "我要投诉，接人工",
    "找客服经理",
    "别机器人回复了，转人工",
    "我要人工处理这个售后",
    "麻烦接入人工",
    "帮我转人工专员",
    "人工客服在吗",
    "需要真人帮我看一下",
]

CANNOT_HANDLE_UTTERANCES = [
    "帮我破解别人账号",
    "把商家后台密码发给我",
    "给我生成一个假发票",
    "帮我删掉别人的订单记录",
    "我要绕过支付直接发货",
    "帮我查一下其他用户的手机号",
    "把数据库导出来给我",
    "给我一个内部管理员权限",
    "帮我刷好评",
    "我要投诉但是编一个不存在的证据",
    "帮我改低商品价格再下单",
    "给我偷看别人的地址",
]

CLARIFY_UTTERANCES = [
    "这个帮我处理一下",
    "有问题",
    "不对",
    "帮我看一下",
    "那个订单怎么回事",
    "它坏了",
    "我要弄一下",
    "这个要怎么搞",
    "处理一下吧",
    "我也不知道怎么说",
]

CANCEL_FLOW_UTTERANCES = [
    "取消当前流程",
    "算了，不办了",
    "退出这个操作",
    "先停止修改",
    "不要继续了",
    "流程取消",
    "回到一开始",
    "我不想继续这个申请",
]


def command_for_slot(slot: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"set slot {slot} {str(value).lower()}"
    if value is False:
        return f"set slot {slot} false"
    if value is None:
        return f"set slot {slot} null"
    if isinstance(value, (int, float)):
        return f"set slot {slot} {value}"
    return f'set slot {slot} "{value}"'


def make_row(
    *,
    user_message: str,
    intent: str,
    commands: list[str],
    category: str,
    context: dict[str, Any] | None = None,
    slots: dict[str, Any] | None = None,
    needs_clarification: bool = False,
) -> dict[str, Any]:
    context = context or {"active_flow": None, "requested_slot": None, "known_slots": {}}
    target = {
        "intent": intent,
        "category": category,
        "commands": commands,
        "dsl": "\n".join(commands),
        "slots": slots or {},
        "needs_clarification": needs_clarification,
    }
    return {
        "task": TASK,
        "source": SOURCE,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "input": user_message,
        "context": context,
        "target": target,
        "completion": json.dumps(target, ensure_ascii=False, sort_keys=True),
    }


def build_examples() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for intent, utterances in FLOW_UTTERANCES.items():
        for user_message in utterances:
            rows.append(
                make_row(
                    user_message=user_message,
                    intent=intent,
                    category="flow_start",
                    commands=[f"start flow {intent}"],
                )
            )

    for case in SLOT_CASES:
        for value, utterances in case["values"]:
            for user_message in utterances:
                context = {
                    "active_flow": case["active_flow"],
                    "requested_slot": case["slot"],
                    "known_slots": {},
                }
                rows.append(
                    make_row(
                        user_message=user_message,
                        intent=case["intent"],
                        category="slot_fill",
                        commands=[command_for_slot(case["slot"], value)],
                        context=context,
                        slots={case["slot"]: value},
                    )
                )

    for user_message in KNOWLEDGE_UTTERANCES:
        rows.append(
            make_row(
                user_message=user_message,
                intent="knowledge_answer",
                category="knowledge",
                commands=["knowledge_answer"],
            )
        )

    for user_message in CHITCHAT_UTTERANCES:
        rows.append(
            make_row(
                user_message=user_message,
                intent="chitchat",
                category="boundary",
                commands=["chitchat"],
            )
        )

    for user_message in HUMAN_HANDOFF_UTTERANCES:
        rows.append(
            make_row(
                user_message=user_message,
                intent="human_handoff",
                category="boundary",
                commands=["human_handoff"],
            )
        )

    for user_message in CANNOT_HANDLE_UTTERANCES:
        rows.append(
            make_row(
                user_message=user_message,
                intent="cannot_handle",
                category="boundary",
                commands=["cannot_handle"],
            )
        )

    for user_message in CLARIFY_UTTERANCES:
        rows.append(
            make_row(
                user_message=user_message,
                intent="clarify",
                category="boundary",
                commands=["clarify"],
                needs_clarification=True,
            )
        )

    for user_message in CANCEL_FLOW_UTTERANCES:
        rows.append(
            make_row(
                user_message=user_message,
                intent="cancel_flow",
                category="boundary",
                commands=["cancel flow"],
                context={
                    "active_flow": "modify_order_receive_info",
                    "requested_slot": "modify_content",
                    "known_slots": {"order_id": "1"},
                },
            )
        )

    # Add paraphrase noise and polite prefixes without changing labels.
    expanded: list[dict[str, Any]] = []
    prefixes = ["", "麻烦", "请", "帮忙", "客服你好，"]
    suffixes = ["", "谢谢", "可以吗", "尽快处理", "我有点着急"]
    for row in rows:
        expanded.append(row)
        if row["target"]["category"] in {"flow_start", "knowledge", "boundary"}:
            for prefix in prefixes[1:3]:
                copy = json.loads(json.dumps(row, ensure_ascii=False))
                copy["input"] = f"{prefix}{row['input']}"
                copy["messages"][-1]["content"] = copy["input"]
                expanded.append(copy)
        if row["target"]["category"] == "slot_fill":
            for suffix in suffixes[1:3]:
                copy = json.loads(json.dumps(row, ensure_ascii=False))
                copy["input"] = f"{row['input']}，{suffix}"
                copy["messages"][-1]["content"] = copy["input"]
                expanded.append(copy)

    return dedupe(expanded)


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = json.dumps(
            {
                "input": row["input"],
                "context": row["context"],
                "commands": row["target"]["commands"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def split_rows(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_intent[row["target"]["intent"]].append(row)

    splits = {"train": [], "val": [], "test": []}
    for intent_rows in by_intent.values():
        rng.shuffle(intent_rows)
        total = len(intent_rows)
        val_count = max(1, round(total * 0.1))
        test_count = max(1, round(total * 0.1))
        train_count = max(1, total - val_count - test_count)

        splits["train"].extend(intent_rows[:train_count])
        splits["val"].extend(intent_rows[train_count : train_count + val_count])
        splits["test"].extend(intent_rows[train_count + val_count :])

    for split_name, split_rows_ in splits.items():
        rng.shuffle(split_rows_)
        for row in split_rows_:
            row["split"] = split_name

    counter = 1
    for split_name in ["train", "val", "test"]:
        for row in splits[split_name]:
            row["id"] = f"cp-{counter:06d}"
            counter += 1

    return splits


def dataset_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Customer Agent Command Parser Example",
        "type": "object",
        "required": ["id", "split", "task", "source", "messages", "input", "context", "target", "completion"],
        "properties": {
            "id": {"type": "string"},
            "split": {"enum": ["train", "val", "test"]},
            "task": {"const": TASK},
            "source": {"const": SOURCE},
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["role", "content"],
                    "properties": {
                        "role": {"enum": ["system", "user", "assistant"]},
                        "content": {"type": "string"},
                    },
                },
            },
            "input": {"type": "string"},
            "context": {
                "type": "object",
                "properties": {
                    "active_flow": {"type": ["string", "null"]},
                    "requested_slot": {"type": ["string", "null"]},
                    "known_slots": {"type": "object"},
                },
            },
            "target": {
                "type": "object",
                "required": ["intent", "category", "commands", "dsl", "slots", "needs_clarification"],
                "properties": {
                    "intent": {"type": "string"},
                    "category": {"type": "string"},
                    "commands": {"type": "array", "items": {"type": "string"}},
                    "dsl": {"type": "string"},
                    "slots": {"type": "object"},
                    "needs_clarification": {"type": "boolean"},
                },
            },
            "completion": {"type": "string"},
        },
    }


def stats_for(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    all_rows = [row for rows in splits.values() for row in rows]
    return {
        "total": len(all_rows),
        "by_split": {split: len(rows) for split, rows in splits.items()},
        "by_intent": Counter(row["target"]["intent"] for row in all_rows),
        "by_category": Counter(row["target"]["category"] for row in all_rows),
        "source": SOURCE,
        "task": TASK,
    }


def readme_for(stats: dict[str, Any]) -> str:
    return f"""# 云枢 Command Parser 微调数据集

这是一个用于微调轻量命令解析模型的合成数据集。它的目标不是训练模型直接回复客户，而是训练模型把用户最新消息和当前对话状态转换成云枢项目已有的命令 DSL。

## 适用任务

```text
用户最新消息 + 当前 Flow / Slot 状态
        ↓
intent / slots / commands
        ↓
Flow / Policy / Action 执行业务逻辑
```

## 文件

- `train.jsonl`：训练集
- `val.jsonl`：验证集
- `test.jsonl`：测试集
- `schema.json`：样本字段说明
- `stats.json`：样本统计

## LLaMA-Factory 导出

按尚硅谷微调教程继续做 LoRA SFT 时，先执行：

```bash
python scripts/export_command_parser_llamafactory.py
```

脚本会生成 `datasets/command_parser/llamafactory/`，里面包含：

- LLaMA-Factory 可读取的 OpenAI-style `messages` JSONL。
- `dataset_info.snippet.json`。
- Qwen3-0.6B LoRA 训练和合并导出 YAML 配置。

## 样本规模

- 总样本数：{stats["total"]}
- Split：{json.dumps(stats["by_split"], ensure_ascii=False)}
- Intent：{json.dumps(stats["by_intent"], ensure_ascii=False, sort_keys=True)}
- Category：{json.dumps(stats["by_category"], ensure_ascii=False, sort_keys=True)}

## 样本格式

每行是一个 JSON object，核心字段：

- `messages`：可直接转成 chat SFT 格式。
- `context`：当前对话状态，例如 `active_flow` 和 `requested_slot`。
- `target.commands`：云枢现有命令 DSL，例如 `start flow cancel_order`。
- `target.dsl`：多命令时用换行拼接。
- `completion`：已序列化的 target JSON，可直接作为 SFT 输出文本。

## 使用建议

建议先用这个数据集微调 Qwen 0.5B / 1.5B / 3B Instruct 的 LoRA，让小模型只负责输出结构化命令。线上执行仍交给云枢现有的 Flow / Policy / Action，避免小模型直接操作业务状态。

评估时重点看：

- JSON 合法率
- intent accuracy
- slot F1
- command exact match
- Flow 执行成功率
"""


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def generate(output_dir: Path, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_examples()
    splits = split_rows(rows, seed)

    for split_name, split_rows_ in splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", split_rows_)

    all_rows = [row for split_name in ["train", "val", "test"] for row in splits[split_name]]
    write_jsonl(output_dir / "all.jsonl", all_rows)

    stats = stats_for(splits)
    (output_dir / "schema.json").write_text(
        json.dumps(dataset_schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(readme_for(stats), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets") / "command_parser",
        help="Directory where JSONL files and metadata will be written.",
    )
    parser.add_argument("--seed", type=int, default=20260701)
    args = parser.parse_args()

    stats = generate(args.output_dir, args.seed)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
