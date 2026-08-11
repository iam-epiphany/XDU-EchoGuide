"""
文档解析层 —— RAG 知识库导入的统一文件解析入口。

支持格式：
  - .txt / .md：整个文件作为一篇文档（文件名作为标题）
  - .json：数组格式 [{"title", "content", ...}, ...]
  - .pdf：逐页提取文本，返回 page_offsets 供切块时标注页码
  - .docx：按文档流顺序提取段落与表格（表格行以 | 拼接，保留结构）

设计取舍（参考 LangChain 文档加载器思路）：
  - 解析与切分解耦：解析器只负责「文件 → 结构化文档」，切分在 knowledge_base 中做
  - 纯函数、无副作用，方便单元测试
  - 扫描件（无文本层的 PDF）不做 OCR，明确报错提示，避免静默导入空文档
"""
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# 支持的扩展名（上传接口与知识库投放目录共用）
SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".pdf", ".docx"}

# page_offsets 元素: (start, end, page)，start/end 为页文本在全文中的字符偏移
PageOffset = Tuple[int, int, int]


def parse_document(filename: str, data: bytes) -> List[Dict[str, Any]]:
    """
    解析单个文件为知识库文档列表。

    返回格式：[{"title": ..., "content": ..., "format": ..., "page_offsets"?: ...}, ...]
    PDF 文档额外携带 page_offsets，供 KnowledgeBase 切块时计算每块的页码区间。
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

    if ext == ".pdf":
        text, page_offsets = _extract_pdf(data)
        if not text.strip():
            raise ValueError("PDF 无文本层（可能是扫描件），暂不支持 OCR，请改用可复制的 PDF 或 txt/md 格式")
        return [{"title": title, "content": text, "format": "pdf", "page_offsets": page_offsets}]

    if ext == ".docx":
        text = _extract_docx(data)
        if not text.strip():
            raise ValueError("docx 未提取到文本内容（空文档）")
        return [{"title": title, "content": text, "format": "docx"}]

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


def _extract_pdf(data: bytes) -> Tuple[str, List[PageOffset]]:
    """逐页提取 PDF 文本，同时记录每页在全文中的字符偏移区间。

    页与页之间以 "\n\n" 分隔；偏移区间按拼接后的全文计算，
    供 KnowledgeBase 切块后二分定位每块的起止页码。
    """
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        parts = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        raise ValueError(f"PDF 解析失败: {exc}") from exc

    if not parts:
        raise ValueError("PDF 无文本层（可能是扫描件），暂不支持 OCR，请改用可复制的 PDF 或 txt/md 格式")

    sep = "\n\n"
    full_text = sep.join(parts)
    page_offsets: List[PageOffset] = []
    pos = 0
    for page, part in enumerate(parts, start=1):
        page_offsets.append((pos, pos + len(part), page))
        pos += len(part) + len(sep)
    return full_text, page_offsets


def _extract_docx(data: bytes) -> str:
    """按文档流顺序提取段落与表格（python-docx）。

    表格逐行提取、单元格以 " | " 拼接，避免校历/时间表类文档在纯文本化时
    丢失行列结构。
    """
    import docx
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ValueError(f"docx 解析失败: {exc}") from exc

    body: List[str] = []
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            text = Paragraph(child, document).text.strip()
            if text:
                body.append(text)
        elif child.tag == qn("w:tbl"):
            table = Table(child, document)
            for row in table.rows:
                line = " | ".join(cell.text.strip() for cell in row.cells).strip()
                if line:
                    body.append(line)
    return "\n".join(body)
