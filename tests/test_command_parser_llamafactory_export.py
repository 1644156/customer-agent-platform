# -*- coding: utf-8 -*-

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "generate_command_parser_dataset.py"
EXPORTER = ROOT / "scripts" / "export_command_parser_llamafactory.py"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class CommandParserLlamaFactoryExportTests(unittest.TestCase):
    def test_exporter_writes_openai_style_sharegpt_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_dir = temp_root / "command_parser"
            output_dir = temp_root / "llamafactory"

            subprocess.run(
                [sys.executable, str(GENERATOR), "--output-dir", str(source_dir)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(EXPORTER),
                    "--input-dir",
                    str(source_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            source_rows = _load_jsonl(source_dir / "train.jsonl")
            exported_rows = _load_jsonl(output_dir / "yunshu_command_parser_train.jsonl")
            self.assertEqual(len(exported_rows), len(source_rows))

            source = source_rows[0]
            exported = exported_rows[0]
            self.assertEqual(set(exported), {"messages"})

            messages = exported["messages"]
            self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
            self.assertEqual(messages[0]["content"], source["messages"][0]["content"])
            self.assertIn(json.dumps(source["context"], ensure_ascii=False, sort_keys=True), messages[1]["content"])
            self.assertIn(source["input"], messages[1]["content"])
            self.assertEqual(messages[2]["content"], source["completion"])
            json.loads(messages[2]["content"])

    def test_exporter_writes_dataset_info_and_training_configs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_dir = temp_root / "command_parser"
            output_dir = temp_root / "llamafactory"

            subprocess.run(
                [sys.executable, str(GENERATOR), "--output-dir", str(source_dir)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(EXPORTER),
                    "--input-dir",
                    str(source_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            info = json.loads((output_dir / "dataset_info.snippet.json").read_text(encoding="utf-8"))
            expected_names = {
                "yunshu_command_parser_train",
                "yunshu_command_parser_val",
                "yunshu_command_parser_test",
            }
            self.assertEqual(set(info), expected_names)

            for name in expected_names:
                entry = info[name]
                self.assertEqual(entry["formatting"], "sharegpt")
                self.assertEqual(entry["columns"], {"messages": "messages"})
                self.assertEqual(entry["tags"]["role_tag"], "role")
                self.assertEqual(entry["tags"]["content_tag"], "content")
                self.assertEqual(entry["tags"]["system_tag"], "system")
                self.assertTrue((output_dir / entry["file_name"]).exists())

            train_config = (output_dir / "qwen3_0_6b_lora_sft.yaml").read_text(encoding="utf-8")
            self.assertIn("dataset: yunshu_command_parser_train", train_config)
            self.assertIn("eval_dataset: yunshu_command_parser_val", train_config)

            merge_config = (output_dir / "qwen3_0_6b_merge_lora.yaml").read_text(encoding="utf-8")
            self.assertIn("adapter_name_or_path: saves/qwen3-0.6b/yunshu-command-parser-lora", merge_config)

            predict_config = (output_dir / "qwen3_0_6b_lora_predict.yaml").read_text(encoding="utf-8")
            self.assertIn("do_predict: true", predict_config)
            self.assertIn("eval_dataset: yunshu_command_parser_test", predict_config)


if __name__ == "__main__":
    unittest.main()
