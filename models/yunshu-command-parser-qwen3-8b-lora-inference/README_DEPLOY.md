# 云枢命令解析 LoRA 推理包

这是推理用精简包，只包含 LoRA adapter 和必要说明，不包含 optimizer/scheduler/trainer_state 等训练状态。

## 当前服务器路径

- Base model: /mnt/workspace/LLaMA-Factory/model/Qwen3-8B
- LoRA adapter: /mnt/workspace/customer_agent_platform/models/yunshu-command-parser-qwen3-8b-lora-inference

## 推理建议

- 模板：qwen3_nothink
- 关闭 thinking
- temperature: 0
- do_sample: false
- max_new_tokens: 220

## 输入格式

当前对话状态：
{"active_flow": null, "known_slots": {}, "requested_slot": null}

用户最新消息：
我要取消订单

请只输出目标 JSON，不要解释。

## 输出格式

严格 JSON，例如：
{"category":"flow_start","commands":["start flow cancel_order"],"dsl":"start flow cancel_order","intent":"cancel_order","needs_clarification":false,"slots":{}}
