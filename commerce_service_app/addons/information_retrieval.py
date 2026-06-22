"""
实现思路：
    先确定查询知识图谱的入口节点，根据入口节点和用户意图生成Cypher语句进行查询
    1.LLM根据用户输入，确定需要的入口节点类型以及实体
    2.根据提供的入口节点类型和实体，使用混合检索获取候选入口节点信息
    3.LLM根据用户输入和入口节点信息生成Cypher查询语句
    4.LLM验证生成的Cypher语法、逻辑是否正确，罗列出错误信息
    5.LLM根据用户输入、入口节点、错误信息、先前Cypher语句来生成更正后的Cypher语句
    6.执行Cypher查询，返回查询结果
工作流程:
    用户提问：用户输入查询问题
    标签路由：LLM识别问题涉及的节点类型和实体
    节点检索：使用混合检索找到候选入口节点
    Cypher生成：LLM根据入口节点和问题生成Cypher查询语句
    语句验证：验证Cypher语句的正确性
    语句校正：如有错误则进行校正
    执行查询：在Neo4j中执行Cypher查询
    返回结果：将查询结果返回给用户
"""

import os
import re
import json
import dotenv
import jieba
import logging
import asyncio
from collections import OrderedDict
from typing import Any, Text, Dict, List, Optional
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from neo4j.exceptions import CypherSyntaxError
from neo4j_graphrag.retrievers import HybridRetriever
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_neo4j import Neo4jGraph
from neo4j_graphrag.retrievers.text2cypher import extract_cypher
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 导入 customer_agent 基类
from customer_agent.retrieval.base_retriever import InformationRetrieval, SearchResult

# 配置控制台日志
logger = logging.getLogger("retrieval")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("[%(levelname)s]%(asctime)s: %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# 路由输出定义,使用Pydantic定义数据模型
class RouteItem(BaseModel):
    label: str = Field(
        ...,
        description=(
            "节点类型，只能是 Category1、Category2、Category3、Trademark、"
            "SPU、SKU、Attr、User 之一"
        ),
    )
    entity: str = Field(
        ...,
        description='实体文本。label 为 User 时只填写纯数字用户ID，比如"176"',
    )


class RouteOutput(BaseModel):
    outputs: list[RouteItem]


def get_slot_value(slot_data: Any) -> Any:
    """从 tracker.to_dict() 的槽位结构中取出真实槽位值。"""
    if isinstance(slot_data, dict):
        value = slot_data.get("value")
        return value if value is not None else slot_data.get("initial_value")
    return slot_data


def get_chat_history(tracker_state: dict[str, Any], user_id) -> str:
    """从 tracker state 中提取聊天历史"""
    user_id = get_slot_value(user_id)
    chat_history = []
    if not tracker_state or not tracker_state.get("events"):
        return ""
    for event in tracker_state.get("events"):
        if event.get("event") == "user":
            role = f"user_id={user_id}" if user_id else "user"  # 如果有user_id则为"user_id=xxx"，否则为"user"
            chat_history.append(f"{role}:{(event.get('text') or '').strip()}")
        elif event.get("event") == "bot":
            chat_history.append(f"bot:{(event.get('text') or '').strip()}")

    # 返回最近5条聊天记录，用换行符连接成一个字符串。（奇数条，最后一条为用户最后的提问，前面是成对的问答）
    return "\n".join(chat_history[-5:])


class GraphRAG(InformationRetrieval):
    """继承了 InformationRetrieval 基类"""

    def __init__(self, embeddings=None):
        super().__init__(embeddings)
        self.valid_route_labels = {
            "Category1",
            "Category2",
            "Category3",
            "Trademark",
            "SPU",
            "SKU",
            "Attr",
            "User",
        }
        self._category_route_cache: dict[str, list[RouteItem]] = {}
        self.category_entry_rules = {
            "化妆品": ("Category1", "个护化妆"),
            "美妆": ("Category1", "个护化妆"),
            "个护": ("Category1", "个护化妆"),
            "护肤": ("Category1", "个护化妆"),
            "好吃": ("Category1", "食品饮料"),
            "好吃的": ("Category1", "食品饮料"),
            "吃的": ("Category1", "食品饮料"),
            "美食": ("Category1", "食品饮料"),
        }
        self.category_aliases = {}
        self.user_related_keywords = (
            "推荐",
            "猜你喜欢",
            "适合我",
            "我想",
            "我要",
            "帮我",
            "给我",
            "之前",
            "看过",
            "浏览",
            "买过",
            "购买",
            "收藏",
            "我的",
        )
        self.node_name_properties = {
            "Category1": "category1_name",
            "Category2": "category2_name",
            "Category3": "category3_name",
            "Trademark": "trademark_name",
            "SPU": "spu_name",
            "SKU": "sku_name",
            "Attr": "attr_value",
        }
        self.node_properties = {
            label: {prop}
            for label, prop in self.node_name_properties.items()
        }
        self.relationship_types = {"View"}
        # 入口节点可选标签
        self.optional_label = (
            '- Category1:   一级分类，如"食品饮料"、"家用电器"、"手机"\n'
            '- Category2:   二级分类，如"大家电"、"香水彩妆"\n'
            '- Category3:   三级分类，如"手机"、"香水"、"笔记本"\n'
            '- Trademark:   品牌，如"华为"、"索芙特"、"金沙河"\n'
            '- SPU:         商品名称，如"华为Mate 40 pro"\n'
            '- SKU:         单品名称，如"联想(Lenovo) 拯救者Y9000P 2022 16英寸游戏笔记本电脑 i9-12900H RTX3070Ti 钛晶灰"\n'
            '- Attr:        商品属性值，如"70英寸"、"蓝色"、"非有机食品"\n'
            '- User:        用户ID，如"176"\n'
        )
        # 节点标签路由 prompt
        # PromptTemplate是基础的提示模板类，用于传统的文本生成场景：
        # ChatPromptTemplate是专为聊天模型设计的提示模板类（系统消息、用户消息、AI消息等，适合多轮对话）
        self.route_label_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "你是一个智能检索路由Agent。"
                    "现在根据用户输入判断最可能需要的一个或多个标签以及每个标签对应的实体，作为后续Neo4j查询的入口节点\n"
                    "**注意：如果查询与用户相关，需要将用户信息加入入口节点。**\n"
                    "用户消息可能带有前缀 user_id=数字，例如 user_id=25:手机有哪些推荐。"
                    '遇到这种格式时，如果问题包含"推荐"、"适合我"、"我想"、"之前看过"、"我的"等个性化语义，'
                    '必须输出 {{"label": "User", "entity": "25"}}。\n'
                    "如果问题包含明确的商品类别、品牌、商品名或属性，也要同时输出对应入口节点；"
                    '例如 user_id=25:手机有哪些推荐 应输出 [{{"label": "User", "entity": "25"}}, {{"label": "Category1", "entity": "手机"}}]。\n'
                    '以严格JSON格式输出结果，比如"[{{"label": "SPU", "entity": "iPhone 16 Pro"}}]"。'
                    "可选节点类型:\n{optional_label}"
                ),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )
        # Cypher 生成 prompt
        self.generate_cypher_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "你是一个Cypher专家，正在根据入口节点信息和用户输入，参照schema生成准确无误的Cypher查询语句。"
                    "**注意：查询结果中不可以包含嵌入向量等多余属性**\n"
                    "仅返回Cypher语句。\n"
                    "schema:\n{schema}"
                ),
                HumanMessagePromptTemplate.from_template(
                    "入口节点:\n{entry_nodes}\n\n用户输入:\n{query}\n\nCypher语句:"
                ),
            ]
        )
        # Cypher 验证 prompt
        self.validate_cypher_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "你是一位Cypher审查专家。你的任务是只报告会导致查询失败、明显违反schema、或明显答非所问的问题。\n"
                    "不要过度纠错；如果Cypher已经能合理回答用户问题，应返回空列表[]。\n"
                    "检查规则：\n"
                    "1. 只在Cypher确实缺少必要用户约束时，才报告用户信息问题。"
                    "如果入口节点包含User，且Cypher中已经出现 User 节点、user_id 条件，或从该用户出发的 View/Buy/Collect 等行为路径，则认为已使用用户信息。\n"
                    "2. 只在Cypher确实缺少必要分类/品牌/商品/属性过滤时，才报告入口节点条件问题。"
                    "如果Cypher中已经通过节点属性（如 category1_name: '手机'）、路径约束、或等价变量限制体现入口节点，则认为条件已生效。\n"
                    "3. 关系方向必须严格按schema判断；如果schema允许当前方向或可以由路径方向清楚表达，不要报告方向错误。\n"
                    "4. 对推荐类问题，基于用户浏览、购买、收藏等行为返回相关商品，或按行为次数排序，都是可接受的简单推荐逻辑。"
                    "不要因为没有复杂推荐算法、评分、热度字段就报告错误。\n"
                    "5. 只报告语法错误、未定义变量、关系方向确实错误、必要过滤确实缺失、返回内容明显不能回答问题这些硬错误。\n"
                    '以严格JSON列表输出错误信息，比如["错误1", "错误2"]。如果没有硬错误，必须返回[]。\n'
                    "schema:\n{schema}"
                ),
                HumanMessagePromptTemplate.from_template(
                    "入口节点:\n{entry_nodes}\n\n"
                    "用户输入:\n{query}\n\n"
                    "待验证的Cypher语句:\n{cypher}"
                ),
            ]
        )
        # Cypher 校正 prompt
        self.correct_cypher_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "你是一位Cypher专家，正在审查一位初级开发人员编写的Cypher语句。你需要根据schema以及提供的错误信息更正Cypher语句。"
                    "仅返回Cypher语句。\n"
                    "schema:\n{schema}"
                ),
                HumanMessagePromptTemplate.from_template(
                    "入口节点:\n{entry_nodes}\n\n"
                    "用户输入:\n{query}\n\n"
                    "错误信息:\n{errors}\n\n"
                    "待更正的Cypher语句:\n{cypher}\n\n"
                    "更正后的Cypher语句:"
                ),
            ]
        )

    def _init_embeddings(self):
        """初始化嵌入模型（延迟加载）"""
        if self.embeddings is not None:
            return

        from langchain_core.embeddings import Embeddings
        from sentence_transformers import SentenceTransformer

        model_path = os.getenv("EMBEDDING_MODEL", "./models/bge-base-zh-v1.5")

        class SimpleEmbedding(Embeddings):
            def __init__(self, path):
                self.model = SentenceTransformer(path)

            def embed_query(self, text: str) -> list[float]:
                return self.embed_documents([text])[0]

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                embeddings = self.model.encode(
                    texts, batch_size=64, normalize_embeddings=True
                )
                return [list(map(float, emb)) for emb in embeddings]

        self.embeddings = SimpleEmbedding(model_path)
        logger.info("嵌入模型已加载: %s", model_path)

    def connect(self, config: Optional[Dict[str, Any]] = None) -> None:
        """连接检索系统：连接到Neo4j数据库并初始化相关组件"""
        config = config or {}

        # 1、创建Neo4j驱动程序
        neo4j_url = config.get("uri") or config.get("neo4j_url", "bolt://localhost:7687")
        neo4j_user = config.get("user", "neo4j")
        neo4j_password = config.get("password", "")
        neo4j_auth = (neo4j_user, neo4j_password)
        # Neo4j 驱动
        self.driver = GraphDatabase.driver(neo4j_url, auth=neo4j_auth)

        # 2、获取图数据库schema
        # Neo4j Graph 包装器
        neo4j_graph = Neo4jGraph(
            neo4j_url,
            neo4j_auth[0],  # 连接Neo4j数据库的用户名
            neo4j_auth[1],  # 连接Neo4j数据库的密码
            enhanced_schema=True  # 是否使用增强的schema信息，设为True 提供更详细的schema信息，包括节点标签、关系类型和属性等
        )
        # Neo4j schema
        self.neo4j_schema = neo4j_graph.schema
        structured_schema = neo4j_graph.structured_schema
        self.node_properties = self._extract_node_properties(structured_schema)
        self.relationship_types = {
            rel.get("type")
            for rel in structured_schema.get("relationships", [])
            if rel.get("type")
        }
        # Neo4j 关系列表
        corrector_schema = [
            Schema(el["start"], el["type"], el["end"])
            for el in structured_schema.get("relationships")
        ]

        # 3、初始化 Cypher查询校正器（langchain提供的api）
        self.cypher_corrector = CypherQueryCorrector(corrector_schema)

        # 4、配置 LLM（使用coder模型，对语法处理效果更好）
        # 保持思考模式开启，有助于 Cypher 生成/校验等复杂任务的准确性
        model_name = "qwen3-coder-plus-2025-07-22"
        dotenv.load_dotenv()
        model_api_key = os.getenv("DASHSCOPE_API_KEY")
        self.llm = ChatTongyi(model=model_name, api_key=model_api_key)

        # 5、初始化嵌入模型
        self._init_embeddings()

        logger.info("GraphRAG 已连接到 Neo4j: %s", neo4j_url)

    def _extract_json_from_text(self, text: str) -> str:
        """从文本中提取 JSON 部分。
        
        处理 LLM 输出可能包含的额外内容（如思考过程 <think>...</think>、markdown 代码块等）。
        """
        if not text:
            return "[]"
        
        import re
        text = text.strip()
        
        # 1. 移除思考模式的 <think>...</think> 标签内容
        text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
        
        # 2. 尝试直接解析
        if text.startswith("[") or text.startswith("{"):
            return text
        
        # 3. 尝试从 markdown 代码块中提取
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if code_block_match:
            return code_block_match.group(1).strip()
        
        # 4. 尝试找到 JSON 数组或对象
        json_match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', text)
        if json_match:
            return json_match.group(1)
        
        return "[]"

    def _parse_route_items(self, json_text: str) -> list[RouteItem]:
        """解析路由JSON，兼容数组和 {"outputs": [...]} 两种格式。"""
        try:
            payload = json.loads(json_text)
        except Exception as e:
            logger.warning("入口节点JSON解析失败: %s，原始内容: %s", e, json_text)
            return []

        if isinstance(payload, dict):
            payload = payload.get("outputs", [])
        if not isinstance(payload, list):
            logger.warning("入口节点JSON格式异常，期望列表或outputs对象: %s", payload)
            return []

        route_items = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            entity = str(item.get("entity") or "").strip()
            if not label or not entity:
                continue
            route_items.append(RouteItem(label=label, entity=entity))
        return route_items

    def _normalize_route_label(self, label: str) -> str:
        """归一化LLM可能输出的标签别名。"""
        label = (label or "").strip()
        alias_map = {
            "category1": "Category1",
            "category2": "Category2",
            "category3": "Category3",
            "trademark": "Trademark",
            "brand": "Trademark",
            "spu": "SPU",
            "sku": "SKU",
            "attr": "Attr",
            "attribute": "Attr",
            "user": "User",
            "用户": "User",
        }
        return alias_map.get(label, alias_map.get(label.lower(), label))

    def _extract_node_properties(self, structured_schema: dict[str, Any]) -> dict[str, set[str]]:
        """从 Neo4j 结构化 schema 中提取各节点可用属性，供兜底 Cypher 裁剪使用。"""
        properties = {
            label: {prop}
            for label, prop in self.node_name_properties.items()
        }
        raw_node_props = structured_schema.get("node_props", {})
        if isinstance(raw_node_props, dict):
            for label, props in raw_node_props.items():
                prop_names = properties.setdefault(label, set())
                for prop in props or []:
                    if isinstance(prop, dict) and prop.get("property"):
                        prop_names.add(prop["property"])
                    elif isinstance(prop, str):
                        prop_names.add(prop)
        return properties

    def _node_has_property(self, label: str, property_name: str) -> bool:
        """判断 schema 中节点是否存在某属性，避免 Neo4j property key warning。"""
        return property_name in self.node_properties.get(label, set())

    def _fallback_sku_return_fields(self) -> list[tuple[str, str]]:
        """兜底查询返回字段，只包含 schema 中确实存在的 SKU 属性。"""
        fields = [("sku.sku_name", "sku_name")]
        optional_fields = [
            ("sku.sku_price", "sku_price"),
            ("sku.sku_desc", "sku_desc"),
        ]
        for expression, alias in optional_fields:
            property_name = expression.split(".", 1)[1]
            if self._node_has_property("SKU", property_name):
                fields.append((expression, alias))
        return fields

    def _append_route_item_once(
            self,
            route_items: list[RouteItem],
            label: str,
            entity: Any
    ) -> None:
        """向路由结果追加唯一入口节点。"""
        label = self._normalize_route_label(label)
        entity = str(entity or "").strip()
        if label not in self.valid_route_labels or not entity:
            return

        key = (label, entity.lower())
        exists = {
            (self._normalize_route_label(item.label), str(item.entity).strip().lower())
            for item in route_items
        }
        if key not in exists:
            route_items.append(RouteItem(label=label, entity=entity))

    def _extract_user_id_from_text(self, text: str) -> Optional[str]:
        """从 user_id=25、用户ID=25 等文本中提取用户ID。"""
        if not text:
            return None
        match = re.search(r"(?:user_id|用户ID|用户id)\s*[=:：]\s*([0-9]+)", text)
        return match.group(1) if match else None

    def _should_use_user_context(self, query: str, context_text: str = "") -> bool:
        """判断当前问题是否需要使用用户画像/行为上下文。"""
        text = f"{query or ''}\n{context_text or ''}"
        return any(keyword in text for keyword in self.user_related_keywords)

    def _candidate_category_terms_from_query(self, query: str, limit: int = 12) -> list[str]:
        """从用户文本中生成可能的类目短语，具体是否成立交给图谱节点校验。"""
        text = re.sub(r"user_id\s*[=:：]\s*\d+", " ", query or "", flags=re.IGNORECASE)
        text = re.sub(r"[，。！？、；：,.!?;:()\[\]{}<>《》\"'“”‘’/\\|]", " ", text)

        stopwords = {
            "推荐", "一下", "一个", "一些", "帮我", "给我", "我想", "我要",
            "想买", "买", "看看", "请", "的", "用来", "用于", "预算",
            "以内", "内", "左右", "吧", "吗", "呢", "和", "或者", "以及",
            "比较", "适合", "好用", "好点", "便宜", "贵", "性价比",
            "打游戏", "游戏", "电竞", "办公", "拍照", "摄影", "日常",
        }
        tokens = [
            token.strip()
            for token in jieba.lcut(text)
            if re.fullmatch(r"[a-zA-Z0-9\u4e00-\u9fa5]+", token.strip())
            and token.strip() not in stopwords
            and not token.strip().isdigit()
        ]

        candidates: OrderedDict[str, None] = OrderedDict()
        for size in range(min(4, len(tokens)), 0, -1):
            for index in range(0, len(tokens) - size + 1):
                candidate = "".join(tokens[index:index + size]).strip()
                if len(candidate) < 2 or candidate in stopwords:
                    continue
                candidates[candidate] = None

        for match in re.finditer(r"[\u4e00-\u9fa5A-Za-z0-9]{2,}", text):
            candidate = match.group(0).strip()
            if candidate and candidate not in stopwords and not candidate.isdigit():
                candidates[candidate] = None

        return list(candidates.keys())[:limit]

    def _graph_category_route_items(self, query: str, top_k: int = 3) -> list[RouteItem]:
        """基于图谱已有分类节点补齐类目入口，避免为每个商品类目写规则。"""
        cache_key = (query or "").strip()
        if cache_key in self._category_route_cache:
            return list(self._category_route_cache[cache_key])

        route_items: list[RouteItem] = []
        candidates = self._candidate_category_terms_from_query(query)
        if not candidates or not self.driver:
            return route_items

        for candidate in candidates:
            for label in ("Category3", "Category2", "Category1"):
                try:
                    nodes = self._direct_node_lookup(label, candidate, top_k=top_k)
                except Exception as e:
                    logger.debug(
                        "图谱类目候选匹配失败，label=%s, candidate=%s, error=%s",
                        label,
                        candidate,
                        e,
                    )
                    nodes = []
                for node in nodes:
                    entity = node.get(self.node_name_properties[label])
                    if entity:
                        self._append_route_item_once(route_items, label, entity)
                if route_items:
                    self._category_route_cache[cache_key] = list(route_items)
                    return route_items

        self._category_route_cache[cache_key] = []
        return route_items

    def _graph_category_terms_from_query(self, query: str) -> list[str]:
        """返回图谱动态识别到的分类名，用于兜底检索和语义验收。"""
        terms = [
            str(item.entity).strip()
            for item in self._graph_category_route_items(query)
            if str(item.entity or "").strip()
        ]
        return list(dict.fromkeys(terms))

    def _augment_route_items(
            self,
            route_items: list[RouteItem],
            query: str,
            context_text: str = "",
            user_id: Optional[Any] = None,
    ) -> list[RouteItem]:
        """用确定性规则补齐LLM遗漏的用户和常见分类入口。"""
        normalized_items = []
        for item in route_items:
            self._append_route_item_once(normalized_items, item.label, item.entity)

        user_id = get_slot_value(user_id)
        inferred_user_id = str(user_id).strip() if user_id is not None else None
        if not inferred_user_id:
            inferred_user_id = self._extract_user_id_from_text(context_text or query)
        if inferred_user_id and self._should_use_user_context(query, context_text):
            self._append_route_item_once(normalized_items, "User", inferred_user_id)

        query_text = query or ""
        for keyword, (label, entity) in self.category_entry_rules.items():
            if keyword in query_text:
                self._append_route_item_once(normalized_items, label, entity)
        for item in self._graph_category_route_items(query_text):
            self._append_route_item_once(normalized_items, item.label, item.entity)

        return normalized_items

    def _format_entry_node(self, label: str, node: dict, score: float = 1.0) -> dict:
        """把Neo4j节点转换为入口节点提示词使用的轻量结构。"""
        prop_name = self.node_name_properties[label]
        return {
            prop_name: node.get(prop_name),
            "score": score,
        }

    def _direct_node_lookup(self, label: str, entity: str, top_k: int) -> list[dict]:
        """优先用精确/包含匹配查入口节点，避免简单实体依赖向量索引。"""
        prop_name = self.node_name_properties.get(label)
        if not prop_name:
            return []

        queries = [
            (
                f"MATCH (n:{label}) "
                f"WHERE toString(n.`{prop_name}`) = $entity "
                "RETURN n LIMIT $top_k"
            ),
            (
                f"MATCH (n:{label}) "
                f"WHERE toString(n.`{prop_name}`) CONTAINS $entity "
                "RETURN n LIMIT $top_k"
            ),
        ]
        for cypher in queries:
            records = self.driver.execute_query(
                cypher,
                {"entity": entity, "top_k": top_k},
            ).records
            if records:
                return [
                    self._format_entry_node(label, dict(record["n"]))
                    for record in records
                ]
        return []

    async def route_label(self, query):
        """
        路由标签识别：识别标签，抽取实体
        使用LLM识别用户查询中涉及的节点类型和实体
        """

        # 1、填充prompt中的变量
        prompt = self.route_label_prompt.format_prompt(
            optional_label=self.optional_label, query=query
        )

        # 2、调用LLM，获得输出结果
        # with_structured_output方法：输出遵循指定的数据结构（指定的RouteOutput类）
        # 依赖于 function calling 或 tool calling 机制，LangChain 会将数据模型（如 RouteOutput）转换为工具定义（tool definition）
        try:
            llm_output = await self.llm.with_structured_output(RouteOutput).ainvoke(prompt)
            if llm_output is None:
                logger.info("LLM structured output 返回 None，尝试普通调用方式")
                llm_output = await self.llm.ainvoke(prompt)
                json_str = self._extract_json_from_text(llm_output.content)
                outputs = self._parse_route_items(json_str)
            else:
                outputs = llm_output.outputs or []
        except Exception as e:
            # 如果模型不支持 tool call 或调用失败，使用普通调用方式
            logger.warning(f"LLM structured output 调用失败: {e}，尝试普通调用方式")
            llm_output = await self.llm.ainvoke(prompt)
            json_str = self._extract_json_from_text(llm_output.content)
            outputs = self._parse_route_items(json_str)

        outputs = self._augment_route_items(outputs, query=query, context_text=query)

        logger.info("入口节点标签与实体:%s", outputs)
        return outputs

    async def node_retrieval(self, route_res, top_k):
        """
        节点检索：根据标签和实体，检索入口节点
        对于用户节点直接通过Cypher查询获取，对于其他类型的节点则使用混合检索（向量+全文）进行检索
            route_res: 路由结果，包含标签和实体信息
            top_k: 检索返回的节点数量上限
        """
        pairs = []  # 用于存储需要检索的标签-实体对
        retrieved_nodes = {}  # 用于存储检索到的节点结果，以标签为键

        for i in route_res:
            label = self._normalize_route_label(i.label)
            entity = str(i.entity or "").strip()
            if label not in self.valid_route_labels or not entity:  # 遍历路由结果中的每一项，如果实体为空则跳过当前项
                continue
            if label == "User":  # 如果标签是"User"，则直接使用Cypher查询在数据库中查找用户节点。
                user_id = int(entity) if entity.isdigit() else entity
                user_records = self.driver.execute_query(
                    "match (u:User) where u.user_id = $user_id return u;",
                    {"user_id": user_id},
                ).records
                if not user_records and entity.isdigit():
                    user_records = self.driver.execute_query(
                        "match (u:User) where u.user_id = $user_id return u;",
                        {"user_id": entity},
                    ).records
                user_info = dict(user_records[0]["u"]) if user_records else {"user_id": user_id}
                retrieved_nodes.setdefault(label, []).append(user_info)  # 将结果添加到retrieved_nodes字典中
            else:  # 如果不是用户节点，则将标签和实体作为一个元组添加到pairs列表中，供后续检索使用
                direct_nodes = self._direct_node_lookup(label, entity, top_k)
                if direct_nodes:
                    retrieved_nodes.setdefault(label, []).extend(direct_nodes)
                else:
                    pairs.append((label, entity))

        if not pairs:  # 如果没有需要检索的标签-实体对，直接返回已找到的节点（通常是用户节点）
            return retrieved_nodes

        # 将标签-实体对分离成两个独立的列表：labels和entities
        labels, entities = zip(*pairs)
        labels, entities = list(labels), list(entities)

        # 对每个实体进行中文分词处理，只保留中英文和数字字符，用" OR "连接，生成全文检索查询文本
        query_texts = [
            " OR ".join(
                [
                    word.strip()
                    for word in jieba.lcut(entity)
                    if re.fullmatch(r"[a-zA-Z0-9\u4e00-\u9fa5]+", word.strip())
                ]
            )
            for entity in entities
        ]
        # 对实体进行向量化处理，生成向量表示，用于向量检索
        query_vectors = self.embeddings.embed_documents(entities)

        # 为每个标签创建混合检索任务
        tasks = []
        hybrid_pairs = []
        for label, query_text, query_vector in zip(labels, query_texts, query_vectors):
            try:
                # 创建HybridRetriever实例（neo4j_graphrag库）
                retriever = HybridRetriever(
                    self.driver,
                    vector_index_name=label.lower() + "_vector",  # 向量索引名称
                    fulltext_index_name=label.lower() + "_fulltext",  # 全文索引名称
                )
            except Exception as e:
                logger.warning("%s 入口节点混合检索初始化失败: %s", label, e)
                continue
            hybrid_pairs.append((label, query_text))
            tasks.append(
                # 将同步的检索操作包装为异步任务，以支持并发执行
                asyncio.to_thread(
                    retriever.get_search_results,
                    query_text,
                    query_vector,
                    top_k,
                    effective_search_ratio=2,
                )
            )
        if not tasks:
            logger.info("入口节点:%s", retrieved_nodes)
            return retrieved_nodes
        # 并发执行所有检索任务，并等待所有结果返回。
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理检索结果
        for (label, _), result in zip(hybrid_pairs, results):  # 遍历每一对标签和对应的检索结果
            if isinstance(result, Exception):
                logger.warning("%s 入口节点混合检索失败: %s", label, result)
                continue
            # 根据标签类型构建结果格式，提取节点名称/值和得分，添加到retrieved_nodes字典中
            retrieved_nodes.setdefault(label, []).extend(
                # 对于非"Attr"标签，使用{标签名}_name作为键
                [
                    {
                        f"{label.lower()}_name": i["node"][f"{label.lower()}_name"],
                        "score": i["score"],
                    }
                    for i in result.records
                ]
                if label != "Attr"
                # 对于"Attr"标签，使用{标签名}_value作为键
                else [
                    {
                        f"{label.lower()}_value": i["node"][f"{label.lower()}_value"],
                        "score": i["score"],
                    }
                    for i in result.records
                ]
            )
        logger.info("入口节点:%s", retrieved_nodes)
        return retrieved_nodes

    async def generate_cypher(self, query, entry_nodes):
        """
        Cypher语句生成：生成 Cypher 语句
        用LLM根据入口节点和用户查询生成Cypher查询语句
        """

        # 1、填充prompt中的变量
        prompt = self.generate_cypher_prompt.format_prompt(
            schema=self.neo4j_schema, query=query, entry_nodes=entry_nodes
        )

        # 2、调用LLM
        llm_output = await self.llm.ainvoke(prompt)

        # 3、使用neo4j_graphrag库的extract_cypher函数提取Cypher语句
        cypher = extract_cypher(llm_output.content)

        logger.info("Cypher生成:%s", cypher)
        return cypher

    async def validate_cypher(self, query, entry_nodes, cypher):
        """
        Cypher语句验证：验证 Cypher 语句
        使用 LLM 验证生成的Cypher语句的语法和逻辑正确性
        """

        # 1、验证 Cypher 语法
        errors = []  # 错误列表，用于收集验证过程中发现的错误
        try:
            self.driver.execute_query(f"explain {cypher}")  # 通过explain关键字只检查语法而不实际执行
        except CypherSyntaxError as e:  # 捕获语法错误并添加到错误列表中
            errors.append(str(e))

        # 2、验证 Cypher 逻辑是否符合用户查询意图
        prompt = self.validate_cypher_prompt.format_prompt(
            schema=self.neo4j_schema,
            query=query,
            cypher=cypher,
            entry_nodes=entry_nodes,
        )
        llm_output = await self.llm.ainvoke(prompt)

        try:
            llm_errors = json.loads(llm_output.content)
            if isinstance(llm_errors, list):
                errors.extend(llm_errors)
        except:
            pass
        logger.info("Cypher验证:%s", errors)
        return errors

    async def correct_cypher(self, query, entry_nodes, cypher, errors):
        """
        Cypher语句校正：校正 Cypher 语句
        使用 LLM 进行校正：如果验证失败，则根据错误信息校正Cypher语句。
        """

        # 1、填充prompt中的变量
        prompt = self.correct_cypher_prompt.format_prompt(
            schema=self.neo4j_schema,
            query=query,
            cypher=cypher,
            entry_nodes=entry_nodes,
            errors=errors,
        )

        # 2、调用LLM
        llm_output = await self.llm.ainvoke(prompt)

        # 3、使用neo4j_graphrag库的extract_cypher函数提取Cypher语句
        cypher = extract_cypher(llm_output.content)

        return cypher

    def _validate_cypher_syntax(self, cypher: str) -> list[str]:
        """只用 EXPLAIN 校验 Cypher 是否可执行，不再调用 LLM。"""
        if not cypher or not cypher.strip():
            return ["Cypher为空"]

        try:
            self.driver.execute_query(f"explain {cypher}")
            return []
        except Exception as e:
            return [str(e)]

    async def _repair_cypher_until_valid(
            self,
            query: str,
            entry_nodes: dict,
            cypher: str,
            errors: list[str],
            max_attempts: int = 2,
    ) -> str:
        """修正 Cypher，并在每次修正后重新做语法校验。"""
        current_cypher = cypher
        current_errors = errors

        for attempt in range(max_attempts):
            if not current_errors:
                return current_cypher

            repaired = await self.correct_cypher(
                query,
                entry_nodes,
                current_cypher,
                current_errors,
            )
            if not repaired or not repaired.strip():
                logger.warning("第 %s 次 Cypher修正返回空，停止修正", attempt + 1)
                return ""

            if not self._preserves_user_constraint(current_cypher, repaired):
                logger.warning("第 %s 次 Cypher修正改变了用户约束，拒绝使用修正结果", attempt + 1)
                return ""

            syntax_errors = self._validate_cypher_syntax(repaired)
            if not syntax_errors:
                logger.info("Cypher修正后校验通过:%s", repaired)
                return repaired

            logger.warning(
                "第 %s 次 Cypher修正后仍不可执行:%s",
                attempt + 1,
                syntax_errors,
            )
            current_cypher = repaired
            current_errors = syntax_errors

        return current_cypher

    def _extract_user_ids_from_cypher(self, cypher: str) -> set[str]:
        """提取 Cypher 中显式 user_id 常量，用于防止修正阶段串用户。"""
        if not cypher:
            return set()
        ids = set()
        patterns = [
            r"user_id\s*:\s*['\"]?([0-9]+)['\"]?",
            r"user_id\s*=\s*['\"]?([0-9]+)['\"]?",
        ]
        for pattern in patterns:
            ids.update(re.findall(pattern, cypher, flags=re.IGNORECASE))
        return ids

    def _preserves_user_constraint(self, original: str, repaired: str) -> bool:
        """LLM 可以修语法，但不能把 user_id=1001 修成 user_id=1。"""
        original_user_ids = self._extract_user_ids_from_cypher(original)
        if not original_user_ids:
            return True
        repaired_user_ids = self._extract_user_ids_from_cypher(repaired)
        return repaired_user_ids == original_user_ids

    def _apply_direction_correction(self, cypher: str) -> str:
        """应用关系方向校正；校正器返回空或不可执行时保留原语句。"""
        try:
            corrected_cypher = self.cypher_corrector(cypher)
        except Exception as e:
            logger.warning("Cypher关系方向校正异常，保留校正前语句: %s", e)
            return cypher

        if not corrected_cypher or not corrected_cypher.strip():
            logger.warning(
                "Cypher关系方向校正返回空，保留校正前语句:%s",
                cypher,
            )
            return cypher

        syntax_errors = self._validate_cypher_syntax(corrected_cypher)
        if syntax_errors:
            logger.warning(
                "Cypher关系方向校正结果不可执行，保留校正前语句。errors=%s, corrected=%s",
                syntax_errors,
                corrected_cypher,
            )
            return cypher

        logger.info("Cypher校正:%s", corrected_cypher)
        return corrected_cypher

    def _records_to_search_results(self, records) -> list[SearchResult]:
        """将 Neo4j records 转换为 SearchResult。"""
        results = []
        for rec in records:
            record_dict = dict(rec)
            text = self._format_record_as_text(record_dict)
            source = self._determine_source(record_dict)
            results.append(SearchResult(
                text=text,
                metadata={"source": source, "raw_data": record_dict},
                score=1.0
            ))
        return results

    def _is_product_recommendation_query(self, query: str) -> bool:
        """识别需要商品推荐语义约束的问题，避免无类目时返回任意商品。"""
        query = query or ""
        recommendation_keywords = (
            "推荐", "买什么", "哪个好", "哪款", "适合", "帮我选", "给我选",
            "有什么", "有哪些", "想买", "想要",
        )
        return any(keyword in query for keyword in recommendation_keywords)

    def _category_term_matches(self, term: str, category: str) -> bool:
        """判断一个类目词是否属于某个规范类目或其别名集合。"""
        term = str(term or "").strip().lower()
        category = str(category or "").strip()
        if not term or not category:
            return False
        aliases = self.category_aliases.get(category, ())
        return term == category.lower() or any(term == alias.lower() for alias in aliases)

    def _are_category_terms_compatible(self, left: str, right: str) -> bool:
        """判断两个类目词是否可视为同一商品范围，防止入口节点误召回扩宽查询。"""
        left = str(left or "").strip()
        right = str(right or "").strip()
        if not left or not right:
            return False
        if left == right:
            return True
        if left in right or right in left:
            return True

        for category in self.category_aliases:
            if self._category_term_matches(left, category) and self._category_term_matches(right, category):
                return True
        return False

    def _merge_category_terms(
            self,
            query_category_terms: list[str],
            entry_category_terms: list[str],
    ) -> list[str]:
        """合并类目词；用户显式类目优先，入口节点只补充兼容的更细粒度类目。"""
        if not query_category_terms:
            return list(dict.fromkeys(term for term in entry_category_terms if term))

        merged = list(dict.fromkeys(term for term in query_category_terms if term))
        for term in entry_category_terms:
            if any(self._are_category_terms_compatible(term, query_term) for query_term in query_category_terms):
                merged.append(term)
        return list(dict.fromkeys(term for term in merged if term))

    def _category_match_terms(self, category_terms: list[str]) -> list[str]:
        """展开类目词为可用于结果验收的同义词集合。"""
        match_terms = []
        for term in category_terms:
            term = str(term or "").strip()
            if not term:
                continue
            match_terms.append(term)
            for category, aliases in self.category_aliases.items():
                if self._category_term_matches(term, category):
                    match_terms.append(category)
                    match_terms.extend(aliases)
        return list(dict.fromkeys(str(term).lower() for term in match_terms if str(term).strip()))

    def _result_matches_category_terms(
            self,
            result: SearchResult,
            category_terms: list[str],
    ) -> bool:
        """校验检索结果是否仍属于用户请求的商品类目。"""
        if not category_terms:
            return True

        raw_data = result.metadata.get("raw_data", {}) if result.metadata else {}
        raw_values = raw_data.values() if isinstance(raw_data, dict) else []
        result_text = " ".join(
            str(value)
            for value in [result.text, *raw_values]
            if value is not None
        ).lower()
        if not result_text.strip():
            return False

        return any(term in result_text for term in self._category_match_terms(category_terms))

    def _filter_semantic_results(
            self,
            query: str,
            results: list[SearchResult],
            entry_nodes: Optional[dict] = None,
            category_terms: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """过滤答非所问的结果；无法证明同类时交给上层降级，而不是硬答。"""
        if not results:
            return []

        if category_terms is None:
            query_terms, _ = self._build_fallback_terms(query)
            entry_terms = self._fallback_category_terms_from_entry_nodes(entry_nodes or {})
            category_terms = self._merge_category_terms(query_terms, entry_terms)

        if not category_terms:
            if self._is_product_recommendation_query(query):
                logger.warning("推荐类查询缺少明确类目，拒绝返回宽泛商品结果: %s", query)
                return []
            return results

        filtered = [
            result
            for result in results
            if self._result_matches_category_terms(result, category_terms)
        ]
        if len(filtered) != len(results):
            logger.warning(
                "检索结果语义验收过滤 %s/%s 条，category_terms=%s",
                len(results) - len(filtered),
                len(results),
                category_terms,
            )
        return filtered

    def _build_fallback_terms(self, query: str) -> tuple[list[str], list[str]]:
        """从用户问题中提取商品类目词和偏好词。"""
        category_terms = self._graph_category_terms_from_query(query)
        preference_terms = []

        preference_rules = {
            "photo": (
                ("拍照", "摄影", "照片", "相机", "影像", "摄像", "人像", "夜景"),
                ["拍照", "摄影", "相机", "影像", "摄像", "像素", "AI拍照", "人像", "夜景"],
            ),
            "gaming": (
                ("游戏", "打游戏", "电竞", "手游", "性能"),
                ["游戏", "电竞", "性能", "高刷", "散热", "DPI", "鼠标", "机械", "12GB", "16GB"],
            ),
            "cosmetics": (
                ("化妆品", "美妆", "个护", "护肤", "彩妆", "香水", "口红", "面霜", "润唇膏"),
                ["化妆品", "美妆", "个护", "护肤", "彩妆", "香水", "口红", "面霜", "润唇膏"],
            ),
            "food": (
                ("好吃", "好吃的", "吃的", "食品", "饮料", "零食", "保健食品", "美食", "小吃"),
                ["食品", "饮料", "零食", "美食", "小吃", "保健食品"],
            ),
        }
        for triggers, terms in preference_rules.values():
            if any(trigger in query for trigger in triggers):
                preference_terms.extend(terms)

        return list(dict.fromkeys(category_terms)), list(dict.fromkeys(preference_terms))

    def _fallback_category_terms_from_entry_nodes(self, entry_nodes: dict) -> list[str]:
        """从入口节点中提取分类名，保证兜底不丢失 LLM 已定位的类目。"""
        category_terms = []
        for label in ("Category1", "Category2", "Category3"):
            prop_name = self.node_name_properties[label]
            for item in entry_nodes.get(label, []) if isinstance(entry_nodes, dict) else []:
                if not isinstance(item, dict):
                    continue
                value = item.get(prop_name)
                if value:
                    category_terms.append(str(value).strip())
        return list(dict.fromkeys(term for term in category_terms if term))

    def _run_fallback_product_query(
            self,
            user_id: Optional[Any],
            category_terms: list[str],
            preference_terms: list[str],
            use_user: bool,
            require_preference: bool,
            top_k: int,
    ):
        """执行确定性商品兜底查询。"""
        params = {
            "user_id": int(user_id) if str(user_id).isdigit() else user_id,
            "user_id_text": str(user_id) if user_id is not None else "",
            "category_terms": category_terms,
            "preference_terms": preference_terms,
            "top_k": top_k,
        }

        sku_text_parts = ["coalesce(toString(sku.sku_name), '')"]
        if self._node_has_property("SKU", "sku_desc"):
            sku_text_parts.append("coalesce(toString(sku.sku_desc), '')")
        search_text_parts = sku_text_parts + [
            "coalesce(toString(spu.spu_name), '')",
            "coalesce(toString(tm.trademark_name), '')",
            "coalesce(toString(c3.category3_name), '')",
            "coalesce(toString(c2.category2_name), '')",
            "coalesce(toString(c1.category1_name), '')",
        ]
        search_text_expr = " + ' ' + ".join(search_text_parts)
        return_fields = ", ".join(
            f"{expression} AS {alias}"
            for expression, alias in self._fallback_sku_return_fields()
        )
        category_return_fields = (
            "c1.category1_name AS category1_name, "
            "c2.category2_name AS category2_name, "
            "c3.category3_name AS category3_name"
        )
        return_fields = f"{return_fields}, {category_return_fields}"
        behavior_relationships = [
            rel for rel in ("View", "Buy", "Collect")
            if rel in self.relationship_types
        ]
        if use_user and not behavior_relationships:
            logger.warning("schema中不存在可用用户行为关系，跳过用户兜底检索")
            return []
        behavior_pattern = "|".join(behavior_relationships)

        graph_context = (
            "OPTIONAL MATCH (sku)-[:Belong]->(spu:SPU) "
            "OPTIONAL MATCH (spu)-[:Belong]->(tm:Trademark) "
            "OPTIONAL MATCH (spu)-[:Belong]->(c3:Category3)-[:Belong]->(c2:Category2)-[:Belong]->(c1:Category1) "
            "WITH sku, c1, c2, c3, "
            f"toLower({search_text_expr}) AS search_text "
        )
        category_filter = (
            "size($category_terms) = 0 OR "
            "any(term IN $category_terms WHERE "
            "search_text CONTAINS toLower(term) OR "
            "c1.category1_name = term OR "
            "c2.category2_name = term OR "
            "c3.category3_name = term)"
        )
        preference_filter = (
            "size($preference_terms) = 0 OR "
            "any(term IN $preference_terms WHERE search_text CONTAINS toLower(term))"
        )

        where_parts = [category_filter]
        if require_preference:
            where_parts.append(preference_filter)
        where_clause = " AND ".join(f"({part})" for part in where_parts)

        if use_user:
            cypher = (
                "MATCH (u:User) "
                "WHERE u.user_id = $user_id OR toString(u.user_id) = $user_id_text "
                f"MATCH (u)-[:{behavior_pattern}]->(sku:SKU) "
                f"{graph_context}"
                f"WHERE {where_clause} "
                f"RETURN DISTINCT {return_fields} "
                "ORDER BY sku_name "
                "LIMIT $top_k"
            )
        else:
            cypher = (
                "MATCH (sku:SKU) "
                f"{graph_context}"
                f"WHERE {where_clause} "
                f"RETURN DISTINCT {return_fields} "
                "ORDER BY sku_name "
                "LIMIT $top_k"
            )

        logger.info("执行确定性兜底Cypher:%s", cypher)
        return self.driver.execute_query(cypher, params).records

    async def _fallback_product_search(
            self,
            query: str,
            user_id: Optional[Any],
            top_k: int,
            entry_nodes: Optional[dict] = None,
    ) -> list[SearchResult]:
        """LLM Cypher 失败或查空时，执行更保守的商品兜底检索。"""
        query_category_terms, preference_terms = self._build_fallback_terms(query)
        entry_category_terms = self._fallback_category_terms_from_entry_nodes(entry_nodes or {})
        category_terms = self._merge_category_terms(query_category_terms, entry_category_terms)

        if not category_terms:
            logger.info("兜底检索缺少明确类目，拒绝执行宽泛商品检索")
            return []

        attempts = []
        if user_id:
            attempts.extend([
                (True, True),
                (True, False),
            ])
        if preference_terms:
            attempts.append((False, True))
        attempts.append((False, False))

        for use_user, require_preference in attempts:
            try:
                records = self._run_fallback_product_query(
                    user_id=user_id,
                    category_terms=category_terms,
                    preference_terms=preference_terms,
                    use_user=use_user,
                    require_preference=require_preference,
                    top_k=top_k,
                )
                results = self._records_to_search_results(records)
                results = self._filter_semantic_results(
                    query,
                    results,
                    entry_nodes=entry_nodes,
                    category_terms=category_terms,
                )
                if results:
                    logger.info(
                        "确定性兜底检索命中 %s 条，use_user=%s, require_preference=%s",
                        len(results),
                        use_user,
                        require_preference,
                    )
                    return results
            except Exception as e:
                logger.warning(
                    "确定性兜底检索失败，use_user=%s, require_preference=%s, error=%s",
                    use_user,
                    require_preference,
                    e,
                )

        return []

    def _format_record_as_text(self, record: dict) -> str:
        """将 Cypher 查询结果记录格式化为自然语言文本。
        
        将形如 {'s.sku_name': 'xxx', 'tm.trademark_name': 'yyy'} 的结果
        转换为易于 LLM 理解的文本格式。
        """
        # 字段名映射：将 Cypher 返回的别名映射为友好的中文名
        field_name_map = {
            # SKU 相关
            'sku_name': '商品名称',
            's.sku_name': '商品名称',
            'sku_price': '价格',
            's.sku_price': '价格',
            'sku_desc': '商品描述',
            's.sku_desc': '商品描述',
            # SPU 相关
            'spu_name': 'SPU名称',
            'sp.spu_name': 'SPU名称',
            # 品牌相关
            'trademark_name': '品牌',
            'tm.trademark_name': '品牌',
            # 分类相关
            'category1_name': '一级分类',
            'c1.category1_name': '一级分类',
            'category2_name': '二级分类',
            'c2.category2_name': '二级分类',
            'category3_name': '三级分类',
            'c3.category3_name': '三级分类',
            # 属性相关
            'attr_name': '属性名',
            'a.attr_name': '属性名',
            'attr_value': '属性值',
            'av.attr_value': '属性值',
            # 用户相关
            'user_id': '用户ID',
            'u.user_id': '用户ID',
        }
        
        parts = []
        for key, value in record.items():
            # 尝试获取友好的字段名
            friendly_name = field_name_map.get(key)
            if not friendly_name:
                # 如果没有预定义映射，尝试从键名提取
                # 例如 's.sku_name' -> 'sku_name' -> '商品名称'
                clean_key = key.split('.')[-1] if '.' in key else key
                friendly_name = field_name_map.get(clean_key, clean_key)
            
            # 只添加非空值
            if value is not None and str(value).strip():
                parts.append(f"{friendly_name}: {value}")
        
        return "，".join(parts) if parts else str(record)
    
    def _determine_source(self, record: dict) -> str:
        """根据查询结果字段确定数据来源类型。"""
        keys = set(record.keys())
        key_str = str(keys).lower()
        
        if 'sku' in key_str:
            return "商品信息"
        elif 'spu' in key_str:
            return "产品信息"
        elif 'trademark' in key_str:
            return "品牌信息"
        elif 'category' in key_str:
            return "分类信息"
        elif 'attr' in key_str:
            return "属性信息"
        elif 'user' in key_str:
            return "用户信息"
        else:
            return "知识库"

    async def search(
            self,
            query: Text,
            top_k: int = 5,
            tracker_state: Optional[Dict[Text, Any]] = None
    ) -> List[SearchResult]:
        """
        执行查询
        整个检索流程的主入口，按顺序执行各步骤并返回结果。
        """

        # 如果查询为空，则返回空列表
        query = (query or "").strip()
        if not query:
            return []

        # 获取用户ID
        user_id = None
        if tracker_state:
            user_id = get_slot_value(tracker_state.get("slots", {}).get("user_id"))
        # 获取聊天历史
        chat_history = get_chat_history(tracker_state, user_id) if tracker_state else query
        # 获取入口节点标签
        route_res = await self.route_label(chat_history)
        route_res = self._augment_route_items(
            route_res,
            query=query,
            context_text=chat_history,
            user_id=user_id,
        )
        logger.info("增强后的入口节点标签与实体:%s", route_res)
        # 检索入口节点
        entry_nodes = await self.node_retrieval(route_res, 10)
        # 生成 Cypher 语句
        cypher = await self.generate_cypher(query, entry_nodes)
        if not cypher or not cypher.strip():
            logger.warning("Cypher生成结果为空，尝试确定性兜底检索")
            return await self._fallback_product_search(query, user_id, top_k, entry_nodes)

        # 验证 Cypher 语句
        try:
            errors = await self.validate_cypher(query, entry_nodes, cypher)
        except Exception as e:
            logger.warning("Cypher验证异常，改用语法校验结果继续处理: %s", e)
            errors = self._validate_cypher_syntax(cypher)

        # 校正 Cypher 语句
        if errors:
            cypher = await self._repair_cypher_until_valid(query, entry_nodes, cypher, errors)

        if not cypher or not cypher.strip():
            logger.warning("Cypher生成/校正结果为空，尝试确定性兜底检索")
            return await self._fallback_product_search(query, user_id, top_k, entry_nodes)

        syntax_errors = self._validate_cypher_syntax(cypher)
        if syntax_errors:
            logger.warning("Cypher修正后仍不可执行，尝试确定性兜底检索: %s", syntax_errors)
            return await self._fallback_product_search(query, user_id, top_k, entry_nodes)

        # 校正关系方向。如果某个关系和其反向关系都不合法，会返回空字符串
        cypher = self._apply_direction_correction(cypher)

        # 执行 Cypher 语句
        try:
            records = self.driver.execute_query(cypher).records
            results = self._records_to_search_results(records)
            results = self._filter_semantic_results(query, results, entry_nodes=entry_nodes)
        except Exception as e:
            logger.warning("执行Cypher语句异常，尝试确定性兜底检索: %s", e)
            results = []

        if not results:
            fallback_results = await self._fallback_product_search(query, user_id, top_k, entry_nodes)
            if fallback_results:
                logger.info("LLM Cypher无结果，返回确定性兜底结果: %s", fallback_results)
                return fallback_results

        logger.info("检索结果: %s", results)
        return results

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j 连接已关闭")


if __name__ == "__main__":
    # 检索测试
    import asyncio
    from dotenv import load_dotenv
    load_dotenv()

    neo4j_url = os.getenv("NEO4J_URL", "bolt://127.0.0.1:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if not neo4j_password:
        raise RuntimeError("请先设置 NEO4J_PASSWORD 环境变量")


    async def test_retrieval(query):
        """测试检索过程"""
        graphrag = GraphRAG()
        graphrag.connect({
            "uri": neo4j_url,
            "user": neo4j_user,
            "password": neo4j_password,
        })
        results = await graphrag.search(
            query,
            top_k=5,
            tracker_state={
                "slots": {"user_id": "25"},
                "events": [{"event": "user", "text": query}],
            },
        )
        print(f"检索结果: {len(results)} 条")
        for i, r in enumerate(results):
            print(f"{i+1}. {r.text[:200] if len(r.text) > 200 else r.text}")
        graphrag.close()


    query = "手机有哪些商品？"
    # query = "白色256GB的手机有哪些？"
    # query = "非有机的大米有哪些，都是什么品牌的？"
    # query = "我想找一款16英寸左右，32G内存2TB硬盘的笔记本，屏幕要求2.5K以上"
    # query = "我之前看到过一款平板电视还不错，我记得是70多寸8K的，能帮我找下是哪个吗"
    # query = "有没有带保湿功能的润唇膏，都是什么品牌的，帮我详细介绍下"
    # query = "帮我推荐oppo的手机"
    asyncio.run(test_retrieval(query))
