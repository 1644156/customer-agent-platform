# 云枢客服系统

云枢是一个面向电商售前、售中、售后场景的智能客服 Agent 原型。项目将 LLM 意图理解、YAML Flow、SQLAlchemy Action、MySQL 业务数据、Neo4j 商品图谱、云问 MCP 文档问答和调试页面整合到一个可演示的客服系统中。

> 项目定位：AI 应用后端工程化原型，重点展示任务型 Agent、业务流程编排、知识源路由和测试意识。它不是生产环境直接上线版本。

## Demo

本仓库保留核心代码、启动配置、示例配置、测试和演示截图。

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

## 解决的问题

电商客服不是单纯聊天，它同时包含确定性的业务流程和开放式知识问答：

- 订单查询、取消订单、修改收货信息需要明确流程和数据库操作。
- 物流查询、售后申请需要槽位收集、状态推进和业务 Action。
- 商品咨询、商品推荐、文档问答需要检索增强。
- 闲聊、无法处理、异常情况需要降级和兜底。

如果全部交给大模型自由生成，业务状态不可控；如果全部写成硬编码规则，扩展成本又很高。

云枢采用的核心思路是：

```text
LLM 负责理解用户意图
Flow / Policy / Action 负责可控执行业务
RAG / GraphRAG / MCP 负责开放知识补充
```

## 架构概览

```text
用户消息
   |
   v
FastAPI / Chat UI
   |
   v
LangGraph 消息处理图
   |-- understand: LLM 输出结构化命令
   |-- policy: 选择 Flow 或知识策略
   |-- action: 执行业务 Action
   |-- guard: 防止单轮流程死循环
   v
response: 汇总回复

确定性业务
   |-- YAML Flow
   |-- SQLAlchemy Action
   |-- MySQL: 订单 / 地址 / 物流 / 售后

开放式知识
   |-- YunwenMcpRetriever -> 云问文档知识库
   |-- GraphRAG -> Neo4j 商品图谱 / 用户行为图谱
```

## 核心能力

### Agent 主链路

入口：`customer_agent/agent/graph/builder.py`

```text
understand
-> policy
-> action
-> guard
-> response
```

职责：

- `understand`：解析用户消息，调用 LLM 生成结构化命令。
- `policy`：根据当前对话状态选择下一步动作。
- `action`：执行内置 Action 或业务自定义 Action。
- `guard`：限制单轮最大动作数，避免流程死循环。
- `response`：汇总本轮回复。

### LLM 命令化

项目没有让模型直接修改业务状态，而是让模型输出可解释、可检查的命令，再交给 Command、Policy 和 Action 执行。

典型命令：

- `start flow <flow_name>`
- `set slot <slot_name> <value>`
- `knowledge_answer`
- `chitchat`
- `cannot_handle`
- `human_handoff`

### YAML Flow 业务流程

目录：`commerce_service_app/data/flows/`

已沉淀 7 个业务流程：

- `switch_user_id`：切换用户。
- `query_order_detail`：查询订单详情。
- `modify_order_receive_info`：修改收货信息。
- `cancel_order`：取消订单。
- `query_logistics_companys`：查询快递公司。
- `query_shipping_order_logistics`：查询物流信息。
- `apply_postsale`：申请售后。

Flow 用来描述槽位收集、条件跳转和动作执行，降低新增业务流程时的硬编码成本。

### 业务 Action

目录：`commerce_service_app/actions/`

已实现 14 个业务 Action，覆盖：

- 订单列表选择。
- 订单详情查询。
- 收货地址选择和修改。
- 订单取消。
- 快递公司查询。
- 物流详情查询。
- 售后资格检查。
- 售后原因收集。
- 售后申请创建。

业务数据访问使用 SQLAlchemy 和 MySQL 示例数据。

### 知识源路由

核心模块：

- `FlowPolicy`：推进订单、物流、售后等确定性业务流程。
- `EnterpriseSearchPolicy`：处理知识问答、商品推荐、闲聊和无法处理场景。
- `HybridKnowledgeRetriever`：区分商品推荐、文档问答和业务任务。
- `YunwenMcpRetriever`：将文档类问题转发到云问知识库平台。
- `GraphRAG`：使用 Neo4j 商品图谱处理商品咨询和推荐。

这里需要特别区分：

```text
MySQL：保存订单、物流、售后等强事务业务数据。
Neo4j：保存商品、品牌、类目、属性、用户行为关系，用于商品咨询和推荐。
```

## 技术栈

- Python 3.12+
- FastAPI
- LangGraph
- LangChain
- SQLAlchemy
- MySQL
- Neo4j
- GraphRAG / Cypher
- FastMCP
- YAML
- DashScope/Qwen
- pytest

## 项目结构

```text
customer_agent_platform/
├─ customer_agent/
│  ├─ agent/                # Agent、LangGraph 主链路、Action 基类
│  ├─ api/                  # FastAPI 服务与调试页面
│  ├─ cli/                  # customer_agent 命令行
│  ├─ core/                 # Domain、Slot、Tracker、TrackerStore
│  ├─ dialogue_understanding/
│  │  ├─ commands/          # 命令 DSL
│  │  ├─ generator/         # LLM 命令生成和解析
│  │  └─ flow/              # Flow 加载和执行
│  ├─ policies/             # FlowPolicy、EnterpriseSearchPolicy
│  └─ retrieval/            # 检索器接口
├─ commerce_service_app/
│  ├─ actions/              # 电商业务 Action
│  ├─ addons/               # GraphRAG、云问 MCP、混合检索
│  ├─ data/flows/           # 业务 Flow
│  ├─ domain/               # 领域、槽位、响应配置
│  └─ models/README.md      # 本地模型说明，权重不入仓库
├─ docker/                  # MySQL、Neo4j 依赖服务
├─ tests/                   # 自动化测试
├─ .gitignore
└─ pyproject.toml
```

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

依赖服务：

```text
MySQL: 3306
Neo4j: 7474 / 7687
```

### 4. 配置校验

```bash
cd ..
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

两个项目组合后形成：

```text
云枢客服系统：对话编排 + 业务 Flow + 订单/物流/售后 Action
云问文档智能平台：文档导入 + RAG 检索 + CRAG + MCP 工具
```

当用户提出订单、物流、售后问题时，云枢通过 Flow 和 Action 处理。

当用户提出文档知识类问题时，云枢通过 `YunwenMcpRetriever` 调用云问的 `yunwen_query_knowledge_base` 工具。

关键路径：

- 云问 MCP 调用：`commerce_service_app/addons/yunwen_mcp_retriever.py`
- 混合知识路由：`commerce_service_app/addons/hybrid_knowledge_retriever.py`
- 文档知识策略：`customer_agent/policies/enterprise_search_policy.py`

## 测试

当前测试覆盖：

- 业务 Action 输出元数据。
- API 页面可用性。
- EnterpriseSearchPolicy 降级和 RAG 行为。
- GraphRAG fallback 和类别约束。
- HybridKnowledgeRetriever 路由。
- Yunwen MCP 返回值映射和异常兜底。

运行：

```bash
pytest tests -q
```

如果本地缺少完整数据库、模型或 API Key，部分端到端能力需要先完成环境配置。已有测试主要用于验证关键路由、策略和适配器逻辑。

## 关键代码路径

- Agent 主图：`customer_agent/agent/graph/builder.py`
- Agent 加载：`customer_agent/agent/agent.py`
- FlowPolicy：`customer_agent/policies/flow_policy.py`
- EnterpriseSearchPolicy：`customer_agent/policies/enterprise_search_policy.py`
- Flow 定义：`commerce_service_app/data/flows/`
- 业务 Action：`commerce_service_app/actions/`
- MySQL 访问：`commerce_service_app/actions/db.py`
- 商品 GraphRAG：`commerce_service_app/addons/information_retrieval.py`
- 云问 MCP 适配：`commerce_service_app/addons/yunwen_mcp_retriever.py`
- 混合知识路由：`commerce_service_app/addons/hybrid_knowledge_retriever.py`
- 测试目录：`tests/`

## 安全与公开说明

公开仓库不包含：

- `.env` 和真实 API Key。
- 本地 embedding 模型权重。
- Neo4j dump 和本地数据库数据。
- 运行日志、缓存、会话状态和 IDE 配置。

需要运行时，请根据 `.env.example` 自行配置本地服务、模型和密钥。

## 项目边界

当前项目仍是工程化原型：

- Demo 数据和本地环境配置较重，完整运行依赖 MySQL、Neo4j、模型和 API Key。
- LLM 命令解析仍依赖文本和 prompt 约束，生产化更适合升级为结构化输出或工具调用。
- 生产级鉴权、审计、监控、限流、数据脱敏和高并发压测尚未完善。
- 当前更适合作为 AI 应用后端原型、工程化原型或面试作品集，而不是直接替换真实客服系统。
