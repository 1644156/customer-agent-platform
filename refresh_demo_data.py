from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SQL_FILE = ROOT / "docker" / "mysql" / "refresh_demo_orders.sql"


def main() -> int:
    sql = SQL_FILE.read_bytes()
    command = [
        "docker",
        "exec",
        "-i",
        "ecs-mysql",
        "mysql",
        "--default-character-set=utf8mb4",
        "-uroot",
        "-p123321",
        "ecs",
    ]
    completed = subprocess.run(command, input=sql, cwd=ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
