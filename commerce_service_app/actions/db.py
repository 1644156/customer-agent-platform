from pathlib import Path
import subprocess

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def _build_database_url_from_env() -> str | None:
    """从标准 MySQL 环境变量拼接连接串。"""
    import os

    password = os.getenv("MYSQL_PASSWORD") or os.getenv("MYSQL_ROOT_PASSWORD")
    if not password:
        return None

    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    database = os.getenv("MYSQL_DATABASE", "ecs")
    username = os.getenv("MYSQL_USER", "root")
    charset = os.getenv("MYSQL_CHARSET", "utf8")
    return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}?charset={charset}"


def _load_database_url() -> str:
    """读取业务数据库连接串。

    优先级：
    1. ECOMMERCE_DB_URL / DATABASE_URL 完整连接串
    2. endpoints.yml database.url
    3. MYSQL_* 环境变量组合
    """
    import os

    env_url = os.getenv("ECOMMERCE_DB_URL") or os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    endpoints_path = Path(__file__).resolve().parents[1] / "endpoints.yml"
    try:
        from customer_agent.shared.config import _resolve_env_vars
        from customer_agent.shared.yaml_loader import read_yaml_file

        endpoints = read_yaml_file(endpoints_path) or {}
        database = _resolve_env_vars(endpoints.get("database", {}))
        url = database.get("url")
        if url:
            return url
    except Exception:
        pass

    env_parts_url = _build_database_url_from_env()
    if env_parts_url:
        return env_parts_url

    raise RuntimeError(
        "未配置业务数据库连接。请设置 ECOMMERCE_DB_URL/DATABASE_URL，"
        "或设置 MYSQL_PASSWORD/MYSQL_ROOT_PASSWORD 等 MYSQL_* 环境变量。"
    )


url = _load_database_url()

# 配置会话工厂
engine = create_engine(url)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


if __name__ == "__main__":

    def export_db_table_class(run=False):
        """将数据库表映射为Python类"""
        if not run:
            return
        output_path = "db_table_class.py"

        cmd = ["python", "-m", "sqlacodegen", url]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)

    export_db_table_class(True)
