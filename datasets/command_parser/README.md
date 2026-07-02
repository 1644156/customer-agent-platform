# 云枢 Command Parser 微调数据集

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

- 总样本数：999
- Split：{"train": 801, "val": 99, "test": 99}
- Intent：{"apply_postsale": 114, "cancel_flow": 24, "cancel_order": 78, "cannot_handle": 36, "chitchat": 45, "clarify": 30, "human_handoff": 36, "knowledge_answer": 66, "modify_order_receive_info": 282, "query_logistics_companys": 54, "query_order_detail": 90, "query_shipping_order_logistics": 54, "switch_user_id": 90}
- Category：{"boundary": 171, "flow_start": 378, "knowledge": 66, "slot_fill": 384}

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
