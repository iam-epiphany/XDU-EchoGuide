"""
文档解析层 —— RAG 知识库导入的统一文件解析入口。

支持格式：
  - .txt / .md：整个文件作为一篇文档（文件名作为标题）
  - .json / .jsonl：数组格式 [{"title", "content", ...}, ...] / 每行一个对象
  - Firecrawl anydoc 支持的全部格式（统一转成 GFM Markdown，标题/表格/列表结构
    完整保留）：doc/docx/docm、ppt/pps/pot/pptx/pptm/ppsx/ppsm、
    xls/xlsx/xlsm/xlsb、odt/ods/odp、rtf、epub、csv、pdf

设计取舍（参考 LangChain 文档加载器思路）：
  - 解析与切分解耦：解析器只负责「文件 → 结构化文档」，切分在 knowledge_base 中做
  - 纯函数、无副作用，方便单元测试
  - 扫描件（无文本层的 PDF）不做 OCR，明确报错提示，避免静默导入空文档
  - anydoc 一律显式传 format（扩展名推导）：无 magic bytes 的格式（如 csv）自动
    检测会失败；且格式与内容不符时 anydoc 可能静默产出垃圾，不能依赖它自检
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# anydoc 支持扩展名 → 规范格式名（对应 firecrawl-anydoc 0.1.8 的 Format Literal）。
# 显式传 format 的原因见模块 docstring：不依赖 anydoc 的内容自检。
_ANYDOC_EXTENSIONS = {
    ".doc": "doc", ".docx": "docx", ".docm": "docx",
    ".ppt": "ppt", ".pps": "ppt", ".pot": "ppt",
    ".pptx": "pptx", ".pptm": "pptx", ".ppsx": "pptx", ".ppsm": "pptx",
    ".xls": "xlsx", ".xlsx": "xlsx", ".xlsm": "xlsx", ".xlsb": "xlsx",
    ".odt": "odt", ".ods": "ods", ".odp": "odp",
    ".rtf": "rtf", ".epub": "epub", ".csv": "csv", ".pdf": "pdf",
}

# 支持的扩展名（上传接口与知识库投放目录共用）
SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".jsonl"} | set(_ANYDOC_EXTENSIONS)


def parse_document(filename: str, data: bytes) -> List[Dict[str, Any]]:
    """
    解析单个文件为知识库文档列表。

    返回格式：[{"title": ..., "content": ..., "format": ..., ...}, ...]
    不支持的扩展名或解析失败时抛出 ValueError（由 API 层转为 400）。
    """
    name = Path(filename)
    ext = name.suffix.lower()
    title = name.stem

    if ext in (".txt", ".md"):
        text = data.decode("utf-8", errors="ignore")
        if not text.strip():
            raise ValueError("文件内容为空")
        return [{"title": title, "content": text, "format": ext.lstrip(".")}]

    if ext == ".json":
        return _parse_json(data)

    if ext == ".jsonl":
        return _parse_jsonl(data)

    fmt = _ANYDOC_EXTENSIONS.get(ext)
    if fmt is not None:
        text = _extract_anydoc(data, fmt)
        if not text.strip():
            raise ValueError("文档未提取到文本内容（空文档）")
        return [{"title": title, "content": text, "format": ext.lstrip(".")}]

    raise ValueError(f"不支持的文件格式: {ext or '（无扩展名）'}，支持: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")


def _parse_json(data: bytes) -> List[Dict[str, Any]]:
    """JSON 数组文档解析（保留原有字段透传，如 domain/source_url）。"""
    try:
        docs = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON 解析失败: {exc}") from exc
    if not isinstance(docs, list):
        raise ValueError("JSON 文件应为数组格式: [{title, content}, ...]")
    for doc in docs:
        if isinstance(doc, dict):
            doc.setdefault("format", "json")
    return docs


def _parse_jsonl(data: bytes) -> List[Dict[str, Any]]:
    """JSONL 逐行对象解析（每行一个 {title, content, ...}，忽略空行）。"""
    docs: List[Dict[str, Any]] = []
    try:
        lines = data.decode("utf-8").splitlines()
        for line_no, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"JSONL 第 {line_no} 行应为对象 {{title, content, ...}}")
            item.setdefault("format", "jsonl")
            docs.append(item)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSONL 第 {line_no} 行解析失败: {exc}") from exc
    return docs


def _extract_anydoc(data: bytes, fmt: str) -> str:
    """Firecrawl anydoc（Rust 引擎）将任意支持格式转为 GFM Markdown。"""
    import anydoc

    try:
        return anydoc.to_markdown_bytes(data, format=fmt)
    except anydoc.UnsupportedError as exc:
        if "OCR is required" in str(exc):
            raise ValueError(
                "PDF 无文本层（可能是扫描件），暂不支持 OCR，请改用可复制的 PDF 或 txt/md 格式"
            ) from exc
        raise ValueError(f"文件格式无法识别或不受支持: {exc}") from exc
    except anydoc.ConvertError as exc:
        # Malformed/Encrypted/MissingPart/ResourceLimit 均继承 ConvertError
        prefix = "PDF 解析失败" if fmt == "pdf" else "文档解析失败"
        raise ValueError(f"{prefix}: {exc}") from exc
