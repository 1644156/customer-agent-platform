# 云枢：电商智能客服 Agent

云枢是一个面向电商售前、售中、售后场景的智能客服 Agent 原型。项目把 LLM 意图理解、YAML 业务流程、SQLAlchemy 业务 Action、MySQL 示例数据、Neo4j 商品图谱、云问 MCP 文档问答和调试页面整合到一个可演示系统中。

这个项目更适合作为 AI 应用后端作品集来阅读：它展示的是如何把大模型能力放进可控的业务流程里，而不是让模型自由生成一切。确定性的订单、物流、售后流程由 Flow / Policy / Action 执行，开放式商品咨询和文档问答由 RAG / GraphRAG / MCP 补充。

## 如果只看 3 分钟

- **项目定位**：电商智能客服 Agent，覆盖订单查询、订单取消、收货信息修改、物流查询、售后申请、商品推荐和文档问答。
- **技术关键词**：FastAPI、LangGraph、LangChain、YAML Flow、SQLAlchemy、MySQL、Neo4j、GraphRAG、FastMCP、DashScope / Qwen。
- **我重点实现的能力**：Agent 主链路、LLM 命令解析、可配置业务 Flow、业务 Action、知识源路由、GraphRAG、云问 MCP 集成和自动化测试。
- **可展示结果**：仓库内保留了 7 张客服场景 demo 截图，README 下方可直接查看。

## 项目亮点

| 能力 | 说明 |
| --- | --- |
| 可控 Agent 流程 | 使用 LangGraph 编排 understand、policy、action、guard、response，避免模型直接修改业务状态。 |
| LLM 命令化 | 模型输出可解释、可检查的命令，再由 Command、Policy、Action 执行。 |
| YAML Flow | 订单、物流、售后等业务流程通过 YAML 描述槽位收集、条件跳转和动作执行。 |
| 业务 Action | 使用 SQLAlchemy 访问 MySQL 示例数据，实现订单、地址、物流、售后等确定性操作。 |
| 知识源路由 | 区分业务任务、商品咨询、商品推荐、文档问答、闲聊和无法处理场景。 |
| GraphRAG | 使用 Neo4j 商品图谱处理商品咨询和推荐。 |
| MCP 集成 | 通过 `YunwenMcpRetriever` 调用云问文档知识库，让客服 Agent 具备文档问答能力。 |
| 测试覆盖 | 覆盖业务 Action、API 页面、知识路由、GraphRAG fallback、MCP 适配等关键路径。 |

## Demo

### 订单查询

![订单查询](docs/assets/Snipaste_2026-06-23_09-03-06.png)

### 订单详情

![订单详情](docs/assets/Snipaste_2026-06-23_09-03-29.png)

### 取消订单

![取消订单](docs/assets/Snipaste_2026-06-23_09-04-00.png)

### 修改收货信息

![修改收货信息](docs/assets/Snipaste_2026-06-23_09-05-10.png)

### 售后申请

![售后申请](docs/assets/Snipaste_2026-06-23_09-05-21.png)

### 商品推荐

![商品推荐](docs/assets/Snipaste_2026-06-23_09-05-34.png)

### 商品知识问答

![商品知识问答](docs/assets/Snipaste_2026-06-23_09-05-46.png)

## 系统架构

```text
用户消息
   |
   v
FastAPI / Chat UI
   |
   v
LangGraph Agent 主链路
   |-- understand: LLM 解析为结构化命令
   |-- policy: 选择业务 Flow 或知识策略
   |-- action: 执行业务 Action / 检索 Action
   |-- guard: 限制单轮动作数，避免流程死循环
   v
response: 汇总回复

确定性业务
   |-- YAML Flow
   |-- SQLAlchemy Action
   |-- MySQL: 订单 / 地址 / 物流 / 售后

开放式知识
   |-- GraphRAG: Neo4j 商品图谱
   |-- YunwenMcpRetriever: 云问文档知识库
```

## 核心模块

| 模块 | 关键路径 | 作用 |
| --- | --- | --- |
| Agent 主图 | `customer_agent/agent/graph/builder.py` | 编排理解、策略、动作、保护和回复节点。 |
| Agent 加载 | `customer_agent/agent/agent.py` | 加载模型、领域配置、策略和运行时上下文。 |
| FlowPolicy | `customer_agent/policies/flow_policy.py` | 推进订单、物流、售后等确定性业务流程。 |
| EnterpriseSearchPolicy | `customer_agent/policies/enterprise_search_policy.py` | 处理知识问答、商品推荐、闲聊和无法处理场景。 |
| Flow 定义 | `commerce_service_app/data/flows/` | 用 YAML 描述业务流程。 |
| 业务 Action | `commerce_service_app/actions/` | 实现订单、地址、物流、售后等业务操作。 |
| GraphRAG | `commerce_service_app/addons/information_retrieval.py` | 使用 Neo4j 商品图谱做商品咨询和推荐。 |
| 云问 MCP 适配 | `commerce_service_app/addons/yunwen_mcp_retriever.py` | 调用云问知识库工具。 |
| 自动化测试 | `tests/` | 验证关键策略、适配器、页面和业务动作。 |

## 已实现业务场景

- 切换用户身份。
- 查询订单列表和订单详情。
- 修改订单收货信息。
- 取消订单。
- 查询快递公司和物流详情。
- 申请售后。
- 商品咨询和商品推荐。
- 调用云问项目完成文档类问答。

## 技术栈

- Python 3.12+
- FastAPI
- LangGraph / LangChain
- SQLAlchemy
- MySQL
- Neo4j
- GraphRAG / Cypher
- YAML
- FastMCP
- DashScope / Qwen
- pytest

## 快速启动

### 1. 安装依赖

```bash
cd customer_agent_platform
uv sync
```

或：

```bash
pip install -e .
```

### 2. 配置环境变量

```bash
cp commerce_service_app/.env.example commerce_service_app/.env
cp docker/.env.example docker/.env
```

按本地环境填写：

```text
DASHSCOPE_API_KEY=...
ECOMMERCE_DB_URL=mysql+pymysql://root:123321@127.0.0.1:3306/ecs?charset=utf8
NEO4J_PASSWORD=...
EMBEDDING_MODEL=commerce_service_app/models/bge-base-zh-v1.5
```

本地 embedding 模型权重不放入 GitHub，需要自行下载到 `commerce_service_app/models/`。

### 3. 启动依赖服务

```bash
cd docker
docker compose up -d
```

默认依赖：

```text
MySQL: 3306
Neo4j: 7474 / 7687
```

### 4. 校验配置

```bash
python -m customer_agent train --dry-run --data commerce_service_app/data/flows --config commerce_service_app/config.yml --domain commerce_service_app/domain
```

### 5. 启动服务

```bash
python -m customer_agent run --model commerce_service_app --host 127.0.0.1 --port 5005
```

常用入口：

```text
聊天页面：http://127.0.0.1:5005/chat
调试页面：http://127.0.0.1:5005/inspect
接口文档：http://127.0.0.1:5005/docs
健康检查：http://127.0.0.1:5005/health
```

## 与云问项目的关系

两个项目组合后形成一套更完整的 AI 应用演示：

```text
云枢：对话编排 + 业务 Flow + 订单 / 物流 / 售后 Action
云问：文档导入 + RAG 检索 + CRAG + MCP 工具
```

当用户提出订单、物流、售后问题时，云枢通过 Flow 和 Action 处理。当用户提出文档知识类问题时，云枢通过 `YunwenMcpRetriever` 调用云问的 `yunwen_query_knowledge_base` 工具。

## 测试

```bash
pytest tests -q
```

当前测试覆盖：

- 业务 Action 输出和元数据。
- API 页面可用性。
- EnterpriseSearchPolicy 的知识路由和降级行为。
- GraphRAG fallback 和类别约束。
- HybridKnowledgeRetriever 路由。
- Yunwen MCP 返回值映射和异常兜底。

说明：完整端到端运行依赖本地数据库、模型和 API Key。缺少这些依赖时，部分端到端能力需要先完成环境配置。

## 公开仓库说明

公开版本保留核心代码、示例配置、测试和演示截图，不包含：

- `.env` 和真实 API Key。
- 本地 embedding 模型权重。
- Neo4j dump 和本地数据库数据。
- 运行日志、缓存、会话状态和 IDE 配置。

当前项目是工程化原型，适合用于展示 AI 应用后端、任务型 Agent、业务流程编排、知识源路由、GraphRAG、MCP 集成和测试意识。生产化还需要补充鉴权、审计、监控、限流、数据脱敏和高并发压测。
