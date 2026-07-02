# LLaMA-Factory 导出文件

这个目录是 `datasets/command_parser/*.jsonl` 的 LLaMA-Factory 训练版本。
每条样本使用 OpenAI-style `messages` 字段，并通过 `dataset_info.snippet.json`
注册为 LLaMA-Factory 的 `sharegpt` 数据集。

## 文件说明

- `yunshu_command_parser_train.jsonl`：训练集
- `yunshu_command_parser_val.jsonl`：验证集
- `yunshu_command_parser_test.jsonl`：离线评估集
- `dataset_info.snippet.json`：复制到 `LLaMA-Factory/data/dataset_info.json` 的片段
- `qwen3_0_6b_lora_sft.yaml`：LoRA SFT 训练配置
- `qwen3_0_6b_lora_predict.yaml`：测试集预测配置
- `qwen3_0_6b_merge_lora.yaml`：LoRA 合并导出配置

## 按教程继续

1. 把 3 个 JSONL 文件复制到 `LLaMA-Factory/data/`。
2. 把 `dataset_info.snippet.json` 里的 3 个条目合并到 `LLaMA-Factory/data/dataset_info.json`。
3. 下载教程里的基座模型：

```bash
modelscope download --model Qwen/Qwen3-0.6B --local_dir model/Qwen3-0.6B
```

4. 在 LLaMA-Factory 目录执行训练：

```bash
llamafactory-cli train qwen3_0_6b_lora_sft.yaml
```

5. 用测试集跑预测：

```bash
llamafactory-cli train qwen3_0_6b_lora_predict.yaml
```

6. 合并导出 LoRA：

```bash
llamafactory-cli export qwen3_0_6b_merge_lora.yaml
```

当前配置使用：

- model_name_or_path: `model/Qwen3-0.6B`
- template: `qwen3_nothink`
- LoRA 输出目录: `saves/qwen3-0.6b/yunshu-command-parser-lora`
- 测试集预测目录: `saves/qwen3-0.6b/yunshu-command-parser-predict`
- 合并模型目录: `saves/qwen3-0.6b/yunshu-command-parser-merged`

如果显存不足，优先把 `per_device_train_batch_size` 改成 1，或者在 WebUI 里使用
QLoRA / 4-bit 量化训练。
