from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> int:
    python = PYTHON if PYTHON.exists() else Path(sys.executable)
    command = [
        str(python),
        "-m",
        "customer_agent",
        "run",
        "--model",
        "commerce_service_app",
        "--host",
        "127.0.0.1",
        "--port",
        "5005",
    ]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
