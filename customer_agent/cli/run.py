# -*- coding: utf-8 -*-
"""
运行服务命令

启动对话服务器。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional, List

import click

from customer_agent.shared.constants import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
)

logger = logging.getLogger(__name__)


@click.command("run", help="运行对话服务")
@click.option(
    "--model", "-m",
    type=click.Path(exists=True),
    default="models",
    help="模型目录或模型文件路径",
)
@click.option(
    "--endpoints",
    type=click.Path(exists=True),
    default=None,
    help="端点配置文件路径",
)
@click.option(
    "--host", "-H",
    type=str,
    default=DEFAULT_SERVER_HOST,
    help="服务器监听地址",
)
@click.option(
    "--port", "-p",
    type=int,
    default=DEFAULT_SERVER_PORT,
    help="服务器监听端口",
)
@click.option(
    "--cors",
    type=str,
    multiple=True,
    default=["*"],
    help="CORS允许的源 (可多次指定)",
)
@click.option(
    "--enable-api/--disable-api",
    default=True,
    help="启用/禁用REST API",
)
@click.option(
    "--enable-inspect/--disable-inspect",
    default=True,
    help="启用/禁用调试页面",
)
@click.option(
    "--channel",
    type=click.Choice(["rest", "socketio", "all"]),
    default="all",
    help="启用的通道类型",
)
@click.pass_context
def run_command(
    ctx: click.Context,
    model: str,
    endpoints: Optional[str],
    host: str,
    port: int,
    cors: tuple,
    enable_api: bool,
    enable_inspect: bool,
    channel: str,
) -> None:
    """运行对话服务。
    
    启动FastAPI服务器，提供对话API和WebSocket接口。
    
    示例:
        customer_agent run
        customer_agent run --model ./models/latest
        customer_agent run --port 8080 --enable-inspect
    """
    verbose = ctx.obj.get("verbose", False)
    debug = ctx.obj.get("debug", False)
    
    click.echo("=" * 50)
    click.echo("CustomerAgent AI - 对话服务")
    click.echo("=" * 50)
    
    model_path = Path(model)

    click.echo(f"模型路径: {model_path.absolute()}")
    click.echo(f"服务地址: http://{host}:{port}")
    click.echo(f"调试页面: {'启用' if enable_inspect else '禁用'}")
    click.echo(f"CORS来源: {list(cors)}")
    click.echo()

    # 加载模型目录下的 .env（优先于项目根目录的 .env）
    # 并切换工作目录到模型目录，使相对路径（如嵌入模型路径）正确解析
    _chdir_done = False
    if model_path.is_dir():
        import os as _os
        model_env = model_path / ".env"
        if model_env.exists():
            import dotenv as _dotenv
            _dotenv.load_dotenv(model_env, override=True)
            if verbose:
                click.echo(f"已加载环境变量: {model_env}")
        _os.chdir(model_path.resolve())
        _chdir_done = True
        if verbose:
            click.echo(f"工作目录: {_os.getcwd()}")

    try:
        # 导入必要模块
        from customer_agent.agent.agent import Agent
        from customer_agent.api.server import CustomerAgentServer

        # 加载Agent
        click.echo("加载Agent...")

        # chdir 后用 "." 加载，未 chdir 则用原始路径
        load_path = "." if _chdir_done else str(model_path)

        if _chdir_done:
            # CWD 已切换到模型目录，直接从当前目录加载
            agent = Agent.load(load_path)
        elif Path(load_path).is_dir():
            # 模型是目录但未 chdir（不应发生，兜底处理）
            agent = Agent.load(load_path)
        else:
            # 模型文件模式（.tar.gz）
            agent = Agent.load(load_path)
        
        click.echo("Agent加载完成")
        
        # 创建服务器
        click.echo("启动服务器...")
        server = CustomerAgentServer(
            agent=agent,
            cors_origins=list(cors),
            enable_inspect=enable_inspect,
        )
        
        click.echo()
        click.echo(f"服务已启动: http://{host}:{port}")
        click.echo(f"API文档: http://{host}:{port}/docs")
        click.echo(f"User chat page: http://{host}:{port}/chat")
        if enable_inspect:
            click.echo(f"调试页面: http://{host}:{port}/inspect")
        click.echo("按 Ctrl+C 停止服务")
        click.echo()
        
        # 运行服务器
        server.run(host=host, port=port)
        
    except KeyboardInterrupt:
        click.echo("\n服务已停止")
    except ImportError as e:
        click.echo(f"导入错误: {e}", err=True)
        if debug:
            raise
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"运行失败: {e}", err=True)
        if debug:
            raise
        raise SystemExit(1)


# 导出
__all__ = ["run_command"]
