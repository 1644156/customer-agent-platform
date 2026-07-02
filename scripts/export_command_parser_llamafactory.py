# -*- coding: utf-8 -*-
"""Export the command-parser dataset for LLaMA-Factory SFT.

The source dataset keeps project metadata such as context, target, and split.
LLaMA-Factory only needs a chat-style ``messages`` field, so this exporter
turns each sample into:

system: command parser role instruction
user: compact dialogue state + latest user message
assistant: serialized target JSON
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
DATASET_PREFIX = "yunshu_command_parser"
DEFAULT_INPUT_DIR = Path("datasets") / "command_parser"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "llamafactory"
DEFAULT_MODEL_PATH = "model/Qwen3-0.6B"
DEFAULT_TEMPLATE = "qwen3_nothink"
DEFAULT_LORA_DIR = "saves/qwen3-0.6b/yunshu-command-parser-lora"
DEFAULT_MERGED_DIR = "saves/qwen3-0.6b/yunshu-command-parser-merged"
DEFAULT_PREDICT_DIR = "saves/qwen3-0.6b/yunshu-command-parser-predict"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def build_user_content(row: dict[str, Any]) -> str:
    context = json.dumps(row["context"], ensure_ascii=False, sort_keys=True)
    return "\n".join(
        [
            "当前对话状态：",
            context,
            "",
            "用户最新消息：",
            row["input"],
            "",
            "请只输出目标 JSON，不要解释。",
        ]
    )


def to_llamafactory_row(row: dict[str, Any]) -> dict[str, Any]:
    system_message = row["messages"][0]["content"]
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": build_user_content(row)},
            {"role": "assistant", "content": row["completion"]},
        ]
    }


def dataset_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }


def build_dataset_info() -> dict[str, Any]:
    return {
        f"{DATASET_PREFIX}_{split}": dataset_entry(f"{DATASET_PREFIX}_{split}.jsonl")
        for split in SPLITS
    }


def train_config(model_path: str, template: str, lora_dir: str) -> str:
    return textwrap.dedent(
        f"""\
        model_name_or_path: {model_path}
        trust_remote_code: true
        template: {template}
        stage: sft
        do_train: true
        finetuning_type: lora
        lora_target: all
        lora_rank: 8
        lora_alpha: 16
        lora_dropout: 0.05
        dataset: {DATASET_PREFIX}_train
        eval_dataset: {DATASET_PREFIX}_val
        cutoff_len: 1024
        max_samples: 100000
        preprocessing_num_workers: 4
        output_dir: {lora_dir}
        overwrite_output_dir: true
        per_device_train_batch_size: 2
        gradient_accumulation_steps: 8
        learning_rate: 1.0e-4
        num_train_epochs: 3.0
        lr_scheduler_type: cosine
        warmup_ratio: 0.1
        logging_steps: 10
        save_steps: 100
        eval_strategy: steps
        eval_steps: 100
        plot_loss: true
        report_to: none
        bf16: true
        """
    )


def merge_config(model_path: str, template: str, lora_dir: str, merged_dir: str) -> str:
    return textwrap.dedent(
        f"""\
        model_name_or_path: {model_path}
        adapter_name_or_path: {lora_dir}
        trust_remote_code: true
        template: {template}
        finetuning_type: lora
        export_dir: {merged_dir}
        export_size: 2
        export_device: cpu
        export_legacy_format: false
        """
    )


def predict_config(model_path: str, template: str, lora_dir: str, predict_dir: str) -> str:
    return textwrap.dedent(
        f"""\
        model_name_or_path: {model_path}
        adapter_name_or_path: {lora_dir}
        trust_remote_code: true
        template: {template}
        stage: sft
        do_predict: true
        finetuning_type: lora
        eval_dataset: {DATASET_PREFIX}_test
        cutoff_len: 1024
        max_samples: 100000
        output_dir: {predict_dir}
        overwrite_output_dir: true
        per_device_eval_batch_size: 1
        predict_with_generate: true
        """
    )


def readme_text(model_path: str, template: str, lora_dir: str, merged_dir: str, predict_dir: str) -> str:
    return textwrap.dedent(
        f"""\
        # LLaMA-Factory 导出文件

        这个目录是 `datasets/command_parser/*.jsonl` 的 LLaMA-Factory 训练版本。
        每条样本使用 OpenAI-style `messages` 字段，并通过 `dataset_info.snippet.json`
        注册为 LLaMA-Factory 的 `sharegpt` 数据集。

        ## 文件说明

        - `{DATASET_PREFIX}_train.jsonl`：训练集
        - `{DATASET_PREFIX}_val.jsonl`：验证集
        - `{DATASET_PREFIX}_test.jsonl`：离线评估集
        - `dataset_info.snippet.json`：复制到 `LLaMA-Factory/data/dataset_info.json` 的片段
        - `qwen3_0_6b_lora_sft.yaml`：LoRA SFT 训练配置
        - `qwen3_0_6b_lora_predict.yaml`：测试集预测配置
        - `qwen3_0_6b_merge_lora.yaml`：LoRA 合并导出配置

        ## 按教程继续

        1. 把 3 个 JSONL 文件复制到 `LLaMA-Factory/data/`。
        2. 把 `dataset_info.snippet.json` 里的 3 个条目合并到 `LLaMA-Factory/data/dataset_info.json`。
        3. 下载教程里的基座模型：

        ```bash
        modelscope download --model Qwen/Qwen3-0.6B --local_dir {model_path}
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

        - model_name_or_path: `{model_path}`
        - template: `{template}`
        - LoRA 输出目录: `{lora_dir}`
        - 测试集预测目录: `{predict_dir}`
        - 合并模型目录: `{merged_dir}`

        如果显存不足，优先把 `per_device_train_batch_size` 改成 1，或者在 WebUI 里使用
        QLoRA / 4-bit 量化训练。
        """
    )


def export(input_dir: Path, output_dir: Path, model_path: str, template: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for split in SPLITS:
        source_rows = load_jsonl(input_dir / f"{split}.jsonl")
        exported_rows = [to_llamafactory_row(row) for row in source_rows]
        write_jsonl(output_dir / f"{DATASET_PREFIX}_{split}.jsonl", exported_rows)
        counts[split] = len(exported_rows)

    (output_dir / "dataset_info.snippet.json").write_text(
        json.dumps(build_dataset_info(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "qwen3_0_6b_lora_sft.yaml").write_text(
        train_config(model_path, template, DEFAULT_LORA_DIR),
        encoding="utf-8",
    )
    (output_dir / "qwen3_0_6b_merge_lora.yaml").write_text(
        merge_config(model_path, template, DEFAULT_LORA_DIR, DEFAULT_MERGED_DIR),
        encoding="utf-8",
    )
    (output_dir / "qwen3_0_6b_lora_predict.yaml").write_text(
        predict_config(model_path, template, DEFAULT_LORA_DIR, DEFAULT_PREDICT_DIR),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        readme_text(model_path, template, DEFAULT_LORA_DIR, DEFAULT_MERGED_DIR, DEFAULT_PREDICT_DIR),
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "dataset_prefix": DATASET_PREFIX,
        "counts": counts,
        "model_name_or_path": model_path,
        "template": template,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    args = parser.parse_args()

    result = export(args.input_dir, args.output_dir, args.model_name_or_path, args.template)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
