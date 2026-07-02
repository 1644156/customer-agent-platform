# -*- coding: utf-8 -*-

import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "command_parser"


def _run_generator(tmp_path: Path) -> Path:
    output_dir = tmp_path / "command_parser"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_command_parser_dataset.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output_dir


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class CommandParserDatasetGeneratorTests(unittest.TestCase):
    def test_command_parser_dataset_generator_writes_stable_splits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = _run_generator(Path(temp_dir))

            train = _load_jsonl(output_dir / "train.jsonl")
            val = _load_jsonl(output_dir / "val.jsonl")
            test = _load_jsonl(output_dir / "test.jsonl")
            all_rows = train + val + test

            self.assertGreaterEqual(len(all_rows), 500)
            self.assertEqual(len({row["id"] for row in all_rows}), len(all_rows))
            self.assertTrue((output_dir / "schema.json").exists())
            self.assertTrue((output_dir / "README.md").exists())
            self.assertTrue((output_dir / "stats.json").exists())

            for split_name, rows in [("train", train), ("val", val), ("test", test)]:
                self.assertTrue(rows)
                self.assertTrue(all(row["split"] == split_name for row in rows))

    def test_command_parser_dataset_rows_match_training_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = _run_generator(Path(temp_dir))
            rows = []
            for split in ["train", "val", "test"]:
                rows.extend(_load_jsonl(output_dir / f"{split}.jsonl"))

            required_intents = {
                "switch_user_id",
                "query_order_detail",
                "modify_order_receive_info",
                "cancel_order",
                "query_logistics_companys",
                "query_shipping_order_logistics",
                "apply_postsale",
                "knowledge_answer",
                "chitchat",
                "cannot_handle",
                "human_handoff",
            }
            intents = {row["target"]["intent"] for row in rows}
            self.assertTrue(required_intents <= intents)

            category_counts = Counter(row["target"]["category"] for row in rows)
            self.assertGreaterEqual(category_counts["flow_start"], 100)
            self.assertGreaterEqual(category_counts["slot_fill"], 100)
            self.assertGreaterEqual(category_counts["boundary"], 50)

            for row in rows:
                self.assertEqual(row["task"], "customer_agent_command_parser")
                self.assertEqual(row["source"], "synthetic_v1")
                self.assertEqual(row["messages"][0]["role"], "system")
                self.assertEqual(row["messages"][-1]["role"], "user")
                self.assertIsInstance(row["context"], dict)
                self.assertIsInstance(row["target"]["commands"], list)
                self.assertTrue(row["target"]["commands"])
                self.assertEqual(row["target"]["dsl"], "\n".join(row["target"]["commands"]))
                self.assertIsInstance(row["target"]["slots"], dict)
                self.assertIsInstance(row["target"]["needs_clarification"], bool)


if __name__ == "__main__":
    unittest.main()
