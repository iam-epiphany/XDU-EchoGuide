"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：语义分块（带 overlap）后存入 ChromaDB（自动生成 Embedding）
  2. 语义检索：根据 query 检索最相关的文档片段，支持相关性阈值与领域过滤
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

分块设计（Markdown 结构感知，见 _chunk_text）：
  - 纯文本（无标题/表格/代码块）回退为递归分隔符切分，行为与原实现一致
  - 带结构文档：标题链注入块首（解决"裸文本块"指代丢失）、标题边界优先成块、
    表格与代码块整体保留不拆散；块长（含注入链）不超过 chunk_size

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。

Embedding 模型（v3 起）：
  - 默认使用 mcp.embeddings 的本地 bge-small-zh-v1.5（ONNX，中文优化，512 维），
    向量在客户端计算后交给 ChromaDB（服务端/本地模式一致）；
  - 本地模型不可用时自动回退旧向量空间（knowledge_base_v2 + ChromaDB 内置
    all-MiniLM-L6-v2），保证服务可用 —— 不同 embedding 模型产生的向量不混存
    在同一 collection（collection 名随向量空间切换）。

检索质量设计（面试点）：
  - 分块采用 LangChain RecursiveCharacterTextSplitter 的中文适配：按
    段落 → 换行 → 句号/叹号/问号/分号 → 逗号 → 空格 → 字符 的优先级递归切分，
    保留语义边界，超长句子也能逐级拆开，不会撑爆单块
  - 分块带 overlap（60 字），避免跨块句子被拦腰截断导致召回不全
  - min_score 相关性阈值：低分噪音不进 prompt，避免误导 LLM
  - domain 元数据过滤：领域问题只检索对应领域的文档片段
  - PDF 等文档经 anydoc 转为 Markdown 入库（结构保留）；旧版 pypdf 解析的
    page_offsets（页码区间标注）仍兼容，见 add_documents
"""
import asyncio
import bisect
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb

from mcp.embeddings import get_embedder

logger = logging.getLogger(__name__)


# 默认知识库文档（西电校园场景常见问题）。导出为模块常量供评测/对比脚本复用
# （evaluation/compare_embedders.py 与运行时导入同一份数据）。
DEFAULT_DOCS = [
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


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    v3 起客户端显式注入本地 bge Embedding（mcp.embeddings.get_embedder），
    模型不可用时回退旧 collection 名 + ChromaDB 默认模型（all-MiniLM-L6-v2），
    保证「向量空间一致」与「服务可用」两者兼得。
    """

    # v3 显式使用 cosine + bge 中文向量。旧 collection 的距离空间/向量维度与
    # 新模型不兼容，不能混查；首次启动把旧文档重新写入 v3 以重新生成索引。
    COLLECTION_NAME = "knowledge_base_v3"               # bge-small-zh-v1.5（当前）
    PREVIOUS_COLLECTION_NAME = "knowledge_base_v2"      # MiniLM cosine（回退/迁移源）
    LEGACY_COLLECTION_NAME = "knowledge_base"           # 最早期 L2 空间（迁移源）
    COLLECTION_METADATA = {
        "hnsw:space": "cosine",
        "description": "西电校园知识库（EchoGuide RAG，cosine，bge）",
    }

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

        # Embedding：本地 bge 模型优先；不可用时回退旧向量空间（chroma 默认模型）。
        # 向量空间随 collection 名绑定，任何路径都不会混写两种模型的向量。
        self._embedding_function = get_embedder()
        self._collection_name = (
            self.COLLECTION_NAME if self._embedding_function is not None
            else self.PREVIOUS_COLLECTION_NAME
        )
        if self._embedding_function is None:
            logger.warning(
                "本地 Embedding 模型不可用，知识库回退 MiniLM 向量空间（%s）",
                self.PREVIOUS_COLLECTION_NAME,
            )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata=self.COLLECTION_METADATA,
            embedding_function=self._embedding_function,
        )
        self._migrate_previous_collections()

        # 如果知识库为空，导入默认文档
        if self._collection.count() == 0:
            try:
                self._load_default_docs()
            except Exception as ex:
                # 本地模式首次写入需下载内置 embedding 模型（~79MB）；
                # 下载失败不应拖垮整个服务启动，知识库后续仍可导入文档使用。
                logger.warning(f"默认知识库导入失败（首次启动可稍后重试）: {ex}")
        try:
            self._load_public_docs()
        except Exception as ex:
            logger.warning(f"版本化公开知识导入失败: {ex}")
        try:
            self._load_docs_directory()
        except Exception as ex:
            logger.warning(f"知识库投放目录导入失败: {ex}")

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "...", "domain": "..."}, ...]
        长文档自动语义分块（每片约 500 字，带 60 字 overlap）。

        旧版 PDF 解析（pypdf）产物可携带 page_offsets:
        [(start, end, page), ...]，切块后每块记录 page_start/page_end 元数据。
        （新版 anydoc 解析不再产出该字段，此分支向后兼容保留。）
        """
        ids, docs, metas = [], [], []

        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            domain  = doc.get("domain") or self._infer_domain(title)
            fmt     = str(doc.get("format") or "text")
            page_offsets = doc.get("page_offsets") or []

            if page_offsets:
                chunks = self._chunk_with_pages(content, page_offsets)
            else:
                chunks = [(c, None, None) for c in self._chunk_text(content)]

            for i, (chunk, page_start, page_end) in enumerate(chunks):
                doc_id = hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                meta = {
                    "title": title,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "domain": domain,
                    "format": fmt,
                    "source_url": str(doc.get("source_url") or ""),
                    "updated_at": str(doc.get("updated_at") or ""),
                    "valid_from": str(doc.get("valid_from") or ""),
                    "source_status": str(doc.get("source_status") or "unverified"),
                    "version": str(doc.get("version") or ""),
                    "scope": str(doc.get("scope") or ""),
                }
                if page_start is not None:
                    meta["page_start"] = str(page_start)
                    meta["page_end"] = str(page_end)
                metas.append(meta)

        if ids:
            # ChromaDB 会自动生成 Embedding
            self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
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
                # cosine 距离的定义为 1-cosine_similarity；显式配置 collection
                # 后该换算才成立。为应对数值误差夹紧到 [0, 1]。
                score = round(max(0.0, min(1.0, 1.0 - float(dist))), 4)
                if score < min_score:
                    continue  # 相关性阈值过滤
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    score,
                    "chunk":    meta.get("chunk_index", 0),
                    "domain":   meta.get("domain", "general"),
                    "format":   meta.get("format", "text"),
                    "page_start": meta.get("page_start", ""),
                    "page_end":   meta.get("page_end", ""),
                    "source_url": meta.get("source_url", ""),
                    "updated_at": meta.get("updated_at", ""),
                    "valid_from": meta.get("valid_from", ""),
                    "source_status": meta.get("source_status", "unverified"),
                    "version": meta.get("version", ""),
                    "scope": meta.get("scope", ""),
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
        return await asyncio.to_thread(
            self.search, query, top_k=top_k, min_score=min_score, domain=domain,
        )

    def _migrate_previous_collections(self) -> None:
        """把旧向量空间的原始文本重写入当前 collection（跨模型需重嵌入）。

        迁移链：
          - 当前为 bge 空间（v3）：MiniLM cosine 空间（v2）有数据则迁移；
            v2 为空时回看最早期 L2 空间（knowledge_base）直接迁入 v3；
          - 当前为回退空间（v2，模型不可用）：沿用原有 knowledge_base → v2 迁移。
        迁移可重复执行：仅当当前 collection 为空时进行，失败不阻断服务启动。
        缓存不调用此逻辑（缓存语义随度量变化，直接冷启动更安全）。
        """
        if self._collection.count() != 0:
            return
        if self._embedding_function is not None:
            for old in (self.PREVIOUS_COLLECTION_NAME, self.LEGACY_COLLECTION_NAME):
                if self._reindex_from(old):
                    return
        else:
            self._reindex_from(self.LEGACY_COLLECTION_NAME)

    def _reindex_from(self, source: str) -> bool:
        """把 source collection 的原始文本重写进当前 collection；无数据返回 False。

        get() 不触发 embedding（原样返回存储文本），upsert 时由当前
        embedding_function 重新生成向量 —— 跨向量空间迁移的正确姿势。
        """
        try:
            legacy = self._client.get_collection(source)
            records = legacy.get(include=["documents", "metadatas"])
            ids = records.get("ids") or []
            docs = records.get("documents") or []
            metas = records.get("metadatas") or []
            if ids and docs:
                self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
                logger.info("知识库已从 %s 重新索引 %d 个片段到 %s",
                            source, len(ids), self._collection_name)
                return True
        except Exception as ex:
            logger.debug("未迁移 %s（可忽略）: %s", source, ex)
        return False

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    # 递归分隔符切分优先级（LangChain RecursiveCharacterTextSplitter 中文适配）：
    # 段落 → 换行 → 句号/叹号/问号/分号 → 逗号 → 空格 → 字符级硬切兜底
    _CHUNK_SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", "，", " ", "")

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 60) -> List[str]:
        """
        Markdown 结构感知分块。

        - 纯文本（无标题/表格/代码块）：回退为递归分隔符切分（段落 > 换行 >
          句号 > 逗号 > 空格 > 字符），带 overlap，行为与原实现一致
        - 带结构文档（anydoc 输出 GFM Markdown）：
            * 标题链注入：每块块首带「文档标题 > 小节标题」链，解决裸文本块
              的指代丢失（"该校/此规定"等）；链长计入 chunk_size 预算
            * 标题是硬边界：标题处强制封块开新块
            * 表格与代码块是原子单元：整体成块不拆散，避免结构破坏
            * 段落粒度成块（不再按句/逗号切），超长段落才递归拆分
        """
        if not text.strip():
            return []
        units = self._parse_markdown_units(text)
        if all(kind == "para" for kind, _ in units):
            # 纯文本：保持原切分行为（回归兼容）
            if len(text) <= chunk_size:
                return [text]
            return [c for c, _, _ in self._split_recursive(text, self._CHUNK_SEPARATORS, chunk_size, overlap)]
        # 结构文档（含短文档）统一走组装：短文档也注入链头/保护原子单元
        return self._assemble_structured_chunks(units, chunk_size, overlap)

    # ── Markdown 结构感知分块 ────────────────────────────────────────────────

    _HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")

    @classmethod
    def _parse_markdown_units(cls, text: str) -> List[Tuple[str, str]]:
        """把 Markdown 文本拆成结构单元序列 (kind, text)。

        kind ∈ {"heading", "table", "code", "para"}：
          - heading：ATX 标题行（^#{1,6} 开头，保留井号用于层级判断）
          - table：连续以 | 开头的行（GFM 表格，含 | --- | 分隔行）
          - code：``` 围栏包裹的代码块
          - para：其余文本按空行分段（与纯文本切分的段落边界一致）
        """
        units: List[Tuple[str, str]] = []
        lines = text.split("\n")
        n = len(lines)
        i = 0
        while i < n:
            line = lines[i]
            if cls._HEADING_RE.match(line):
                units.append(("heading", line.strip()))
                i += 1
            elif line.lstrip().startswith("|"):
                rows = []
                while i < n and lines[i].lstrip().startswith("|"):
                    rows.append(lines[i].strip())
                    i += 1
                units.append(("table", "\n".join(rows)))
            elif line.lstrip().startswith("```"):
                block = [line.strip()]
                i += 1
                while i < n and not lines[i].lstrip().startswith("```"):
                    block.append(lines[i].rstrip())
                    i += 1
                if i < n:  # 闭合围栏（未闭合则吞掉到文末）
                    block.append(lines[i].strip())
                    i += 1
                units.append(("code", "\n".join(block)))
            else:
                buf = []
                while (
                    i < n
                    and lines[i].strip()
                    and not cls._HEADING_RE.match(lines[i])
                    and not lines[i].lstrip().startswith("|")
                    and not lines[i].lstrip().startswith("```")
                ):
                    buf.append(lines[i].rstrip())
                    i += 1
                while i < n and not lines[i].strip():
                    i += 1  # 跳过空行（段落分隔）
                if buf:
                    units.append(("para", "\n".join(buf)))
        return units

    def _assemble_structured_chunks(
        self,
        units: List[Tuple[str, str]],
        chunk_size: int,
        overlap: int,
    ) -> List[str]:
        """结构单元 → 最终分块。

        - 维护标题栈：heading 更新链并强制封块；链头注入下一块块首
        - para 单元按段落粒度贪心合并，放不下时封块（块间带 overlap），
          超长段落递归拆分（拆分片段每片独立成块，均带链头）
        - table / code 原子单元整体成块（不参与合并，不做 overlap）
        """
        chunks: List[str] = []
        chain: List[Tuple[int, str]] = []   # (标题级别, 去井号标题)
        pend: List[str] = []                # 当前块正文（段落列表）
        buf_head = ""                       # 当前块的标题链
        last_tail = ""                      # 上一块正文尾部 overlap 字

        def head_text() -> str:
            return " > ".join(title for _, title in chain)

        def close_block(keep_overlap: bool) -> None:
            nonlocal pend, last_tail
            if not pend:
                return
            body = "\n\n".join(pend)
            chunks.append(f"{buf_head}\n{body}" if buf_head else body)
            # overlap 仅在同一标题链的相邻块间携带（跨标题语义边界不重叠）
            last_tail = body[-overlap:] if keep_overlap and len(body) > overlap else ""
            pend = []

        def emit_atomic(text: str) -> None:
            chunks.append(f"{buf_head}\n{text}" if buf_head else text)

        for kind, unit in units:
            if kind == "heading":
                close_block(keep_overlap=False)
                level = len(unit) - len(unit.lstrip("#"))
                title = unit.lstrip("#").strip()
                while chain and chain[-1][0] >= level:
                    chain.pop()
                chain.append((level, title))
                buf_head = head_text()
                continue

            if kind in ("table", "code"):
                close_block(keep_overlap=False)
                emit_atomic(unit)
                continue

            # para：段落粒度贪心合并
            effective = chunk_size - len(buf_head)
            if pend and len("\n\n".join(pend + [unit])) <= effective:
                pend.append(unit)
                continue

            # 放不下当前块：封块并尝试携带 overlap
            if pend:
                close_block(keep_overlap=True)
            if last_tail:
                if len(last_tail) + 2 + len(unit) <= effective:
                    pend = [last_tail + "\n\n" + unit]
                    last_tail = ""
                    continue
                emit_atomic(last_tail)   # overlap 尾巴单独成块（不丢内容）
                last_tail = ""

            if len(unit) > effective:
                # 超长段落：递归拆分（带 overlap），每片独立成块
                for sub, _, _ in self._split_recursive(
                    unit, self._CHUNK_SEPARATORS, effective, overlap
                ):
                    chunks.append(f"{buf_head}\n{sub}" if buf_head else sub)
            else:
                pend = [unit]

        close_block(keep_overlap=False)
        # 尾部残留的 overlap 尾巴（无后续段落承接）单独成块
        if last_tail:
            chunks.append(f"{buf_head}\n{last_tail}" if buf_head else last_tail)
        return chunks

    def _chunk_with_pages(
        self,
        text: str,
        page_offsets: List[Tuple[int, int, int]],
        chunk_size: int = 500,
        overlap: int = 60,
    ) -> List[Tuple[str, Optional[int], Optional[int]]]:
        """
        PDF 专用分块：切块同时经二分定位每块所在页码区间。

        page_offsets 为 [(start, end, page), ...]（document_parser 产出）。
        返回 [(chunk, page_start, page_end), ...]，跨页块 page_start < page_end。
        """
        if not text.strip():
            return []
        if len(text) <= chunk_size:
            chunks = [(text, 0, len(text))]
        else:
            chunks = self._split_recursive(text, self._CHUNK_SEPARATORS, chunk_size, overlap)

        page_starts = [p[0] for p in page_offsets]
        result: List[Tuple[str, Optional[int], Optional[int]]] = []
        for chunk, start, end in chunks:
            # 偏移 s 属于第 (bisect_right(starts, s) - 1) 页；end 为开区间，取 end-1 定位末页
            page_start = page_offsets[bisect.bisect_right(page_starts, start) - 1][2]
            page_end = page_offsets[bisect.bisect_right(page_starts, end - 1) - 1][2]
            result.append((chunk, page_start, page_end))
        return result

    def _split_recursive(
        self,
        text: str,
        separators: Tuple[str, ...],
        chunk_size: int,
        overlap: int,
        base: int = 0,
    ) -> List[Tuple[str, int, int]]:
        """
        递归分隔符切分（LangChain RecursiveCharacterTextSplitter 思路）。

        - 取优先级最高的、在当前文本中出现的分隔符切分
        - 短段（≤ chunk_size）贪心合并；超长段换更细的分隔符递归处理
        - 返回 [(chunk, start, end)]，start/end 为 chunk 在原始全文中的字符偏移
          （合并片段在原文本中天然连续，chunk 与 text[start:end] 等价），供页码映射
        """
        sep = separators[0]
        if sep == "":
            # 字符级硬切分兜底（全角文本无任何可拆分隔符时使用）
            return [(text[i:i + chunk_size], base + i, base + i + chunk_size) for i in range(0, len(text), chunk_size)]

        spans: List[Tuple[int, int]] = []
        pos, start = 0, 0
        while True:
            idx = text.find(sep, pos)
            if idx == -1:
                spans.append((start, len(text)))
                break
            spans.append((start, idx))
            pos = idx + len(sep)
            start = pos
        spans = [(s, e) for s, e in spans if s < e]  # 去掉空段
        if len(spans) < 2:
            # 该分隔符切不出有效片段（如只有段首一个换行），换下一级
            return self._split_recursive(text, separators[1:], chunk_size, overlap, base)

        good: List[Tuple[int, int]] = []
        final: List[Tuple[str, int, int]] = []
        for s, e in spans:
            if e - s <= chunk_size:
                good.append((s, e))
            else:
                final.extend(self._merge_chunks(text, good, chunk_size, overlap, base))
                good = []
                final.extend(self._split_recursive(text[s:e], separators[1:], chunk_size, overlap, base + s))
        final.extend(self._merge_chunks(text, good, chunk_size, overlap, base))
        return final

    @staticmethod
    def _merge_chunks(
        text: str,
        spans: List[Tuple[int, int]],
        chunk_size: int,
        overlap: int,
        base: int = 0,
    ) -> List[Tuple[str, int, int]]:
        """贪心合并相邻片段（LangChain _merge_splits 思路）。

        片段在原文本中相邻（中间只隔分隔符），合并后仍是连续区间；
        放不下时封块，新块携带上一块尾部 overlap 字保持语义连续。
        """
        chunks: List[Tuple[str, int, int]] = []
        current: Optional[Tuple[int, int]] = None
        for s, e in spans:
            if current is None:
                current = (s, e)
                continue
            if e - current[0] <= chunk_size:
                current = (current[0], e)
                continue
            cs, ce = current
            chunks.append((text[cs:ce], base + cs, base + ce))
            current = (ce - overlap, e)
        if current is not None:
            cs, ce = current
            chunks.append((text[cs:ce], base + cs, base + ce))
        return chunks

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（模块级 DEFAULT_DOCS，供评测/对比脚本复用）。"""
        self.add_documents(DEFAULT_DOCS)
        logger.info(f"已导入默认知识库: {len(DEFAULT_DOCS)} 篇文档")

    def _load_public_docs(self) -> None:
        """加载受版本控制、带来源的公开知识；重复启动不会制造重复切片。"""
        default_dir = Path(__file__).resolve().parents[1] / "data" / "public"
        public_dir = Path(os.getenv("ECHOGUIDE_PUBLIC_DATA_DIR", str(default_dir)))
        source = public_dir / "academic_policies.json"
        if not source.exists():
            return
        documents = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(documents, list):
            raise ValueError("academic_policies.json 顶层必须为数组")
        count = self.add_documents(documents)
        logger.info("已同步版本化公开知识: %d 个片段", count)

    def _load_docs_directory(self) -> None:
        """扫描知识库投放目录（默认 data/knowledge_docs/），自动导入文档。

        把 PDF/Word/txt/md 放进目录即可入库，适合运维批量投放（与上传接口等价）。
        doc_id 由 title+chunk 内容 md5 生成，重复导入天然幂等（upsert 覆盖），
        多次启动不会产生重复切片；单文件解析失败只告警，不阻断其他文件。
        """
        from mcp.document_parser import SUPPORTED_EXTENSIONS, parse_document

        default_dir = Path(__file__).resolve().parents[1] / "data" / "knowledge_docs"
        docs_dir = Path(os.getenv("ECHOGUIDE_KNOWLEDGE_DOCS_DIR", str(default_dir)))
        if not docs_dir.is_dir():
            return
        imported = 0
        for path in sorted(docs_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                documents = parse_document(path.name, path.read_bytes())
                imported += self.add_documents(documents)
                logger.info("知识库投放目录导入: %s", path.name)
            except Exception as ex:
                logger.warning("知识库投放目录文档导入失败 %s: %s", path.name, ex)
        if imported:
            logger.info("已从知识库投放目录共导入 %d 个文档片段", imported)
