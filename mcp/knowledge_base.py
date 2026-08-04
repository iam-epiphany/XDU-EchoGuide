"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：语义分块（带 overlap）后存入 ChromaDB（自动生成 Embedding）
  2. 语义检索：根据 query 检索最相关的文档片段，支持相关性阈值与领域过滤
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。

检索质量设计（面试点）：
  - 分块带 overlap（60 字），避免句子被拦腰截断导致召回不全
  - min_score 相关性阈值：低分噪音不进 prompt，避免误导 LLM
  - domain 元数据过滤：领域问题只检索对应领域的文档片段
"""
import hashlib
import logging
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    ChromaDB 内置了 Embedding 模型（all-MiniLM-L6-v2），
    调用 add() 时自动生成向量，query() 时自动做语义匹配。
    不需要额外调用 Anthropic Embeddings API。
    """

    COLLECTION_NAME = "knowledge_base"

    # 标题 → 领域 的粗粒度映射（导入时写入 metadata，供检索按领域过滤）
    TITLE_DOMAIN_MAP = {
        "校历": "affairs", "选课": "academic", "奖学金": "affairs", "请假": "affairs",
        "穿梭车": "campus_life", "校车": "campus_life", "食堂": "campus_life",
        "餐饮": "campus_life", "宿舍": "campus_life", "图书馆": "campus_life",
        "教务系统": "it_help", "校园网": "it_help", "vpn": "it_help", "邮箱": "it_help",
    }

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
    ):
        # 优先连接独立 ChromaDB 服务（服务端内置 embedding 模型，客户端无需下载）
        self._use_server = False
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 使用服务端时不传 embedding_function，让服务端处理
        # 本地模式时也不传，使用 ChromaDB 默认的（会触发模型下载）
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "西电校园知识库（EchoGuide RAG）"},
        )

        # 如果知识库为空，导入默认文档
        if self._collection.count() == 0:
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "...", "domain": "..."}, ...]
        长文档自动语义分块（每片约 500 字，带 60 字 overlap）。
        """
        ids, docs, metas = [], [], []

        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            domain  = doc.get("domain") or self._infer_domain(title)
            chunks  = self._chunk_text(content, chunk_size=500, overlap=60)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                metas.append({
                    "title": title,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "domain": domain,
                })

        if ids:
            # ChromaDB 会自动生成 Embedding
            self._collection.add(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"知识库导入 {len(ids)} 个文档片段")

        return len(ids)

    @staticmethod
    def _infer_domain(title: str) -> str:
        for key, domain in KnowledgeBase.TITLE_DOMAIN_MAP.items():
            if key in title:
                return domain
        return "general"

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.25,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        语义检索：根据 query 返回最相关的文档片段。

        - min_score：相关性阈值，低于阈值的片段直接丢弃（避免噪音误导 LLM）
        - domain：领域过滤（ChromaDB where 条件），如 "it_help" 只检索 IT 领域片段
        """
        where = {"domain": domain} if domain else None
        from core.tracing import sync_span

        with sync_span("kb_search", query=query[:80], top_k=top_k, domain=domain or ""):
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
            )

        items = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                score = round(1.0 - dist, 4)  # ChromaDB 返回距离，转为相似度
                if score < min_score:
                    continue  # 相关性阈值过滤
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    score,
                    "chunk":    meta.get("chunk_index", 0),
                    "domain":   meta.get("domain", "general"),
                })

        return items

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        min_score = params.get("min_score", 0.25)
        domain = params.get("domain")  # Agent 可传当前领域做过滤
        return self.search(query, top_k=top_k, min_score=min_score, domain=domain)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 60) -> List[str]:
        """
        语义分块（带 overlap）：

        - 按句号/换行切分为句子，贪心拼块
        - 每块末尾补上前一块末尾 overlap 字，避免跨块句子被拦腰截断
        """
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        sentences = [s.strip() for s in text.replace("\n", "。").split("。") if s.strip()]
        chunks: List[str] = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                # 携带前一块尾部 overlap 字，保持语义连续性
                tail = current[-overlap:] if current else ""
                current = tail + sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)
        return chunks

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（西电校园场景常见问题）。"""
        default_docs = [
            {
                "title": "校历与重要时间节点",
                "content": (
                    "西安电子科技大学校历说明（以学校官方校历为准，下文为常见结构）。"
                    "每学年分秋季、春季两个学期，通常秋季学期 8 月底或 9 月初开学，春季学期次年 2 月中下旬开学。"
                    "每学期一般 18-20 个教学周，最后 1-2 周为期末考试周。"
                    "寒暑假、国庆、春节等假期安排以校历为准；选课、退改选、考试安排有明确的起止时间。"
                    "重要节点包括：选课开始/结束、退改选截止、期中考试、期末考试周、成绩发布。"
                    "具体日期请以教务系统和学院通知为准。"
                ),
            },
            {
                "title": "选课指南",
                "content": (
                    "西电选课说明。"
                    "选课通过教务系统进行，登录后进入「选课」模块。"
                    "每学期选课一般分为预选、正选、退改选几个阶段，具体时间见校历与选课通知。"
                    "学生需按培养方案修读必修课，并在学分要求内选修通识课与专业选修课。"
                    "退改选期间可以退课或改选；退改选截止后一般不能再调整。"
                    "选课时应注意先修课程要求和总学分上限/下限。"
                    "选课人数不达标的课程可能被停开，请留意教务系统通知。"
                ),
            },
            {
                "title": "校园穿梭车（校车）",
                "content": (
                    "西电校园穿梭车说明（南北校区往返，时刻以后勤/校车管理最新通知为准）。"
                    "校园穿梭车连接南校区（长安校区）与北校区（太白校区），主要服务有跨校区课程或事务的师生。"
                    "工作日通常在早、中、晚多个时段发车，周末和节假日班次可能减少。"
                    "乘车一般需提前在指定系统或小程序预约，凭校园卡或预约信息乘车。"
                    "发车地点一般为各校区指定乘车点，建议提前 5-10 分钟到达。"
                    "末班车时间、临时调整、天气影响等信息以校车管理通知为准。"
                ),
            },
            {
                "title": "食堂与餐饮",
                "content": (
                    "西电食堂与餐饮说明。"
                    "南校区和北校区各有多个学生食堂，提供大众快餐、风味窗口、清真餐等多种选择。"
                    "食堂一般提供早、中、晚三餐，营业时段大致为早餐 6:30-9:00、午餐 11:00-13:00、晚餐 17:00-19:00，具体以各食堂为准。"
                    "就餐使用校园卡刷卡支付，部分窗口支持移动支付。"
                    "校园卡可在圈存机、手机端或指定服务点充值；遗失后应及时挂失补办。"
                    "如有食品安全或价格问题，可向后勤餐饮管理部门反映。"
                ),
            },
            {
                "title": "宿舍管理",
                "content": (
                    "西电学生宿舍管理说明。"
                    "学生宿舍由学校统一分配，按学院、年级、性别安排楼栋与房间。"
                    "宿舍一般有门禁时间，晚归需登记；外来人员探访需按宿管规定登记。"
                    "水电使用按学校规定执行，部分宿舍实行限额或充值制度，超额需自行充值。"
                    "宿舍设施故障（如水电、网络、家具损坏）可通过后勤报修系统或联系宿管报修。"
                    "宿舍内禁止使用大功率违章电器，注意用电与消防安全。"
                    "调宿、退宿等事宜需向学生宿舍管理中心申请。"
                ),
            },
            {
                "title": "图书馆",
                "content": (
                    "西电图书馆使用说明。"
                    "南校区和北校区均设有图书馆，开放时间一般为工作日全天，考试周通常延长开放，节假日开放时间以公告为准。"
                    "入馆需携带校园卡；自习座位可通过图书馆座位预约系统提前预约，预约后需按时签到，违规可能被暂停预约权限。"
                    "借书凭校园卡办理，每本书有规定借阅期限，可在系统内续借；逾期归还会产生违规记录。"
                    "图书馆提供电子资源（数据库、电子书、期刊），在校内网或通过 VPN 可访问。"
                    "图书馆内需保持安静，遵守阅览区、研讨室等区域的使用规定。"
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
