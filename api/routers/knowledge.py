"""知识库路由：文档导入（文本/文件上传）、统计与检索演示。"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from api import state
from api.deps import require_admin

router = APIRouter(prefix="/knowledge", tags=["知识库"])


class DocInput(BaseModel):
    """单篇文档输入。"""
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=100_000)
    domain: Optional[Literal["academic", "campus_life", "affairs", "it_help", "general"]] = None
    source_url: Optional[str] = Field(default=None, max_length=1000)
    updated_at: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    source_status: Literal["official", "unverified", "sample", "stale"] = "unverified"


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput] = Field(min_length=1, max_length=100)


@router.post("/add")
async def add_knowledge(body: BatchDocInput, _admin=Depends(require_admin)):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字）并存入 ChromaDB，ChromaDB 内置 Embedding 模型自动向量化。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "选课指南", "content": "西电选课通过教务系统进行，分预选、正选、退改选阶段..."},
        {"title": "校园穿梭车", "content": "校园穿梭车连接南校区与北校区，工作日班次较多..."}
      ]
    }
    ```
    """
    if state._kb is None:
        raise HTTPException(503, "知识库未初始化")
    kb = state._kb
    count = await asyncio.to_thread(kb.add_documents, [
        {
            "title": d.title, "content": d.content, "domain": d.domain,
            "source_url": d.source_url, "updated_at": d.updated_at.isoformat() if d.updated_at else "",
            "valid_from": d.valid_from.isoformat() if d.valid_from else "", "source_status": d.source_status,
        } for d in body.documents
    ])
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": kb.doc_count}


@router.post("/upload")
async def upload_knowledge(file: UploadFile = File(...), _admin=Depends(require_admin)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json` / `.jsonl`：JSON 数组 `[{"title": "...", "content": "..."}, ...]` / 每行一个对象
    - 其余格式（pdf/doc/docx/ppt/pptx/xls/xlsx/odt/odp/rtf/epub/csv 等）由
      Firecrawl anydoc 统一转为 GFM Markdown，标题/表格结构完整保留；
      扫描件（无文本层的 PDF）会明确报错

    文件大小限制：10MB
    """
    if state._kb is None:
        raise HTTPException(503, "知识库未初始化")
    kb = state._kb

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    from mcp.document_parser import parse_document
    try:
        docs = parse_document(file.filename or "unknown", content)
    except ValueError as e:
        raise HTTPException(400, str(e))

    count = await asyncio.to_thread(kb.add_documents, docs)
    return {
        "message": f"文件 {file.filename} 导入成功",
        "added_chunks": count,
        "total_chunks": kb.doc_count,
    }


@router.get("/stats")
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    if state._kb is None:
        raise HTTPException(503, "知识库未初始化")
    return {"total_chunks": state._kb.doc_count}
