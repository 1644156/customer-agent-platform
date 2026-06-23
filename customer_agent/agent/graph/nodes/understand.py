# -*- coding: utf-8 -*-
"""
理解节点

负责调用 LLMCommandGenerator 生成命令，并调用 CommandProcessor 处理命令。
这是消息处理流程的第一个核心节点。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from customer_agent.core.tracker import UserMessage
from customer_agent.dialogue_understanding.commands.slot_commands import SetSlotCommand

if TYPE_CHECKING:
    from customer_agent.agent.graph.state import MessageProcessingState
    from customer_agent.dialogue_understanding.commands.base import Command

logger = logging.getLogger(__name__)


def parse_set_slots_payload(payload: str) -> List["Command"]:
    """解析 /SetSlots(slot=value) 格式的 payload。
    
    支持按钮点击时直接解析槽位设置，绕过 LLM 处理。
    
    支持的格式：
    - /SetSlots(order_id=123)
    - /SetSlots(order_id="订单123")
    - /SetSlots(slot1=value1, slot2=value2)
    
    Args:
        payload: 以 /SetSlots( 开头的字符串
        
    Returns:
        SetSlotCommand 列表
    """
    commands: List["Command"] = []
    
    # 提取括号内的内容
    match = re.match(r'/SetSlots\((.+)\)$', payload.strip())
    if not match:
        logger.warning(f"[parse_set_slots_payload] 无法解析 payload: {payload}")
        return commands
    
    content = match.group(1)
    
    # 解析 key=value 对
    # 支持格式: slot=value, slot="value with spaces", slot='value'
    pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\)]+))'
    for m in re.finditer(pattern, content):
        slot_name = m.group(1)
        # 取第一个非空的值（带引号或不带引号）
        slot_value = m.group(2) or m.group(3) or m.group(4)
        
        # 尝试转换数字
        if slot_value.isdigit():
            slot_value = int(slot_value)
        elif slot_value.lower() == 'true':
            slot_value = True
        elif slot_value.lower() == 'false':
            slot_value = False
        
        commands.append(SetSlotCommand(name=slot_name, value=slot_value))
        logger.info(f"[parse_set_slots_payload] 解析槽位: {slot_name}={slot_value}")
    
    return commands


async def understand_node(state: "MessageProcessingState") -> Dict[str, Any]:
    """理解节点：生成命令并处理。
    
    该节点执行以下步骤：
    1. 检测 /SetSlots payload（按钮点击），直接解析绕过 LLM
    2. 将用户输入封装为 UserMessage 并更新 tracker
    3. 调用 LLMCommandGenerator 生成命令
    4. 调用 CommandProcessor 处理命令
    
    Args:
        state: 当前图状态
        
    Returns:
        状态更新字典
    """
    tracker = state["tracker"]
    input_message = state["input_message"]
    metadata = state.get("metadata", {})
    domain = state.get("domain")
    flows = state.get("flows")
    
    command_generator = state.get("_command_generator")
    command_processor = state.get("_command_processor")
    
    logger.info(f"[understand_node] 处理消息: {input_message[:50]}...")

    # 1. 检测 /SetSlots payload（按钮点击，绕过 LLM）
    if input_message.strip().startswith("/SetSlots("):
        logger.warning(f"[understand_node] 检测到 /SetSlots payload: {input_message}")
        commands = parse_set_slots_payload(input_message)
        logger.warning(f"[understand_node] 解析结果: {commands}")

        if commands and command_processor:
            # 创建用户消息（记录原始输入）
            user_message = UserMessage(
                text=input_message,
                sender_id=tracker.sender_id,
                metadata={"payload": True, **metadata},
            )
            tracker.update_with_message(user_message)

            # 直接处理解析出的命令
            process_result = command_processor.process(commands, tracker)

            logger.warning(
                f"[understand_node] payload 处理完成: {process_result.commands_executed} 个命令执行, "
                f"errors={process_result.errors}, "
                f"stack_size={tracker.dialogue_stack.size()}, "
                f"active_flow={tracker.active_flow}"
            )
            
            return {
                "tracker": tracker,
                "current_commands": None,  # payload 不经过 LLM，没有 generation_result
                "process_result": process_result,
                "node_history": state.get("node_history", []) + ["understand"],
            }
    
    # 2. 创建用户消息并更新 tracker
    user_message = UserMessage(
        text=input_message,
        sender_id=tracker.sender_id,
        metadata=metadata,
    )
    tracker.update_with_message(user_message)

    # 2.5 如果正在 collect 一个 categorical 槽位，且用户输入匹配允许值，直接设置槽位（绕过 LLM）
    if domain and domain.slots:
        flow_frame = tracker.dialogue_stack.top_flow_frame()
        logger.warning(
            f"[understand_node] step 2.5: domain={domain is not None}, "
            f"flow_frame={flow_frame is not None}, "
            f"stack_size={tracker.dialogue_stack.size()}, "
            f"active_flow={tracker.active_flow}"
        )
        if flow_frame:
            collecting_slot = getattr(flow_frame, "slot_to_collect", None)
            logger.warning(f"[understand_node] step 2.5: collecting_slot={collecting_slot}")
            if collecting_slot:
                slot_obj = domain.slots.get(collecting_slot)
                if slot_obj:
                    allowed = getattr(slot_obj, "values", None) or getattr(slot_obj, "allowed_values", None)
                    logger.warning(f"[understand_node] step 2.5: allowed={allowed}")
                    if allowed:
                        msg_stripped = input_message.strip()
                        matched_value = None
                        if msg_stripped in allowed:
                            matched_value = msg_stripped
                        else:
                            for val in allowed:
                                if isinstance(val, str) and (msg_stripped in val or val in msg_stripped):
                                    matched_value = val
                                    break
                        if matched_value is not None:
                            logger.warning(
                                f"[understand_node] collect步骤直接匹配: "
                                f"{collecting_slot}={matched_value} (输入: {msg_stripped})"
                            )
                            set_cmd = SetSlotCommand(name=collecting_slot, value=matched_value)
                            process_result = command_processor.process([set_cmd], tracker)
                            return {
                                "tracker": tracker,
                                "current_commands": None,
                                "process_result": process_result,
                                "node_history": state.get("node_history", []) + ["understand"],
                            }
                        else:
                            logger.warning(f"[understand_node] step 2.5: 无匹配 (输入='{msg_stripped}')")
    
    # 初始化结果
    current_commands = None
    events = []
    process_result = None
    
    try:
        # 2. 使用命令生成器生成命令
        if command_generator:
            flows_list = flows.flows if flows else []
            generation_result = await command_generator.generate(
                tracker, domain, flows_list
            )
            current_commands = generation_result
            
            logger.warning(
                f"[understand_node] 生成了 {len(generation_result.commands)} 个命令: "
                f"{[str(c) for c in generation_result.commands]}"
            )

            # (collect步骤的categorical槽位已在步骤2.5中直接处理，此处无需兜底)

            # 3. 使用命令处理器处理命令
            if generation_result.commands and command_processor:
                process_result = command_processor.process(
                    generation_result.commands, tracker
                )
                events = process_result.events
                
                logger.info(
                    f"[understand_node] 处理了 {process_result.commands_executed} 个命令, "
                    f"产生 {len(events)} 个事件, "
                    f"下一动作: {process_result.next_action}"
                )
        else:
            logger.warning("[understand_node] 未配置命令生成器，跳过命令生成")
            
    except Exception as e:
        logger.error(f"[understand_node] 处理失败: {e}")
        return {
            "tracker": tracker,
            "current_commands": None,
            "error": str(e),
            "node_history": state.get("node_history", []) + ["understand"],
        }
    
    return {
        "tracker": tracker,
        "current_commands": current_commands,
        "process_result": process_result,
        "node_history": state.get("node_history", []) + ["understand"],
    }


# 导出
__all__ = ["understand_node"]
