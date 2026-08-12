#!/usr/bin/env python3
"""
package_code.py — 一键打包 EchoGuide 主要代码为 zip，供网页版 AI 分析。

用法:
    python scripts/package_code.py   # 或双击项目根目录 package.bat

产出:
    dist/echoguide-code-YYYYMMDD-HHMMSS.zip
    包内含 _AI_SUMMARY.md（目录结构 + 文件清单 + 行数统计 + 关键入口），
    网页版 AI 打开压缩包先读这份摘要即可定位重点。

排除内容（"主要代码"之外的都去掉）:
    .git / .venv / node_modules / dist / __pycache__ / 缓存与 IDE 目录
    .env（密钥，.env.example 保留）、*.db / *.onnx / *.bin / 图片 / PDF
    assets（README 截图）、_docs_gen（文档生成器）、学习文档（PDF 包装资料）
    frontend 仅保留源码与构建配置，不含 node_modules / dist / package-lock
"""
from __future__ import annotations

import datetime
import os
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "dist"

# ── 排除规则 ──────────────────────────────────────────────────────────────────

EXCLUDE_DIR_NAMES = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".idea", ".zcode", ".ruff_cache", ".mypy_cache",
    "node_modules", "dist", "logs",
    "assets", "_docs_gen", "学习文档",   # 截图 / 文档生成器 / PDF 包装资料
}

EXCLUDE_FILE_NAMES = {
    ".env",                 # 密钥（.env.example 会保留）
    "package-lock.json",    # 依赖锁文件，对代码分析无意义
    "package-lock.yaml",
}

BINARY_SUFFIXES = {
    ".pyc", ".pyo", ".db", ".onnx", ".bin", ".pdf",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
}

TEXT_SUFFIXES = {
    ".py", ".js", ".ts", ".vue", ".jsx", ".tsx", ".html", ".css", ".scss",
    ".md", ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".txt",
    ".sh", ".bat", ".ps1", ".example",
}

# 顶层目录职责速查（写入摘要，帮 AI 快速理解结构）
DIR_NOTES = {
    "api":          "FastAPI 入口：/chat、/mcp、/knowledge、/search、认证、SSE 与评测接口",
    "agents":       "五个领域 Agent、复杂度闸门、Planner/DAG/Executor/Synthesizer",
    "core":         "级联意图识别、领域词表、Skills 与 Trace",
    "tools":        "个人/校园/学业/事务/IT 确定性工具",
    "mcp":          "工具管理器、Agentic RAG、MCP 协议与语义缓存",
    "memory":       "分层记忆 L0 原文 / L1 事实 / L2 场景 / L3 画像 + 上下文卸载",
    "personal":     "个人数据（课表/待办/DDL）SQLite 存储与服务",
    "auth":         "登录与会话",
    "campus":       "校园信息",
    "config":       "Prometheus 抓取配置与告警规则",
    "evaluation":   "离线评测与真实 Demo Benchmark",
    "monitor":      "监控指标采集",
    "skills":       "校园 SOP 动态 Skills",
    "echoguide_guard": "Guard：Prompt 注入检测、限流、审计脱敏",
    "frontend":     "Vue 3 前端源码（src/ + 构建配置，不含 node_modules）",
    "data":         "知识库公开数据（data/public）、演示文档（data/demo_docs）",
    ".github":      "CI 工作流",
}


def collect_files():
    """返回 [(绝对路径, 相对 ROOT 的 posix 路径)]；目录名命中排除规则即整棵剪枝。"""
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIR_NAMES)
        for name in sorted(filenames):
            if name in EXCLUDE_FILE_NAMES:
                continue
            if pathlib.Path(name).suffix.lower() in BINARY_SUFFIXES:
                continue
            full = pathlib.Path(dirpath) / name
            files.append((full, full.relative_to(ROOT).as_posix()))
    return sorted(files, key=lambda f: f[1])


def count_lines(path: pathlib.Path) -> int:
    """统计文本文件行数；二进制/非文本按 0 计。"""
    if not path.name.endswith(tuple(TEXT_SUFFIXES)):
        return 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def build_summary(files, total_lines):
    """生成 _AI_SUMMARY.md：结构树 + 规模 Top 文件 + 关键入口，方便 AI 快速定位。"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top_dirs = {}
    for _, rel in files:
        parts = rel.split("/")
        if len(parts) == 1:
            top_dirs.setdefault("(根目录)", []).append(rel)
        else:
            top_dirs.setdefault(parts[0], []).append(rel)

    L = []
    L.append(f"# EchoGuide 代码包摘要（自动生成于 {now}）")
    L.append("")
    L.append(f"- 文件数：{len(files)}")
    L.append(f"- 代码总行数：{total_lines:,}")
    L.append("")
    L.append("## 顶层结构")
    L.append("")
    L.append("```text")
    L.append("EchoGuide/")
    for name in sorted(top_dirs):
        n = len(top_dirs[name])
        note = DIR_NOTES.get(name, "")
        suffix = f"  {note}" if note else ""
        L.append(f"├─ {name}/  ({n} 个文件){suffix}")
    L.append("```")
    L.append("")
    L.append("## 代码规模 Top 20（按行数，供 AI 优先阅读）")
    L.append("")
    L.append("| 文件 | 行数 |")
    L.append("|---|---:|")
    for full, rel in sorted(files, key=lambda f: -count_lines(f[0]))[:20]:
        if count_lines(full) == 0:
            continue
        L.append(f"| `{rel}` | {count_lines(full):,} |")
    L.append("")
    L.append("## 关键入口（快速定位）")
    L.append("")
    L.append("| 入口 | 位置 | 说明 |")
    L.append("|---|---|---|")
    L.append("| FastAPI 应用 | `api/main.py` | 全部路由；`/mcp` 端点 ~L992、MCP 工具注册 ~L220、SSE `/chat/stream` |")
    L.append("| MCP 协议层 | `mcp/protocol.py` | JSON-RPC 2.0 / Streamable HTTP：initialize / tools/list / tools/call |")
    L.append("| 工具框架 | `mcp/tool_manager.py` | 工具注册、熔断、TTL 缓存、查询改写→并行召回→去重→重排链路 |")
    L.append("| Agentic RAG | `mcp/knowledge_base.py` | ChromaDB 检索；本地 bge Embedding/Rerank 在 `mcp/embeddings.py` |")
    L.append("| 多 Agent | `agents/` | 五个领域 Agent、复杂度闸门、Planner/DAG/Executor/Synthesizer |")
    L.append("| 意图识别 | `core/` | 级联意图识别（Pattern→Embedding→LLM）与 Fast/Deep 复杂度闸门 |")
    L.append("| 个人数据 | `personal/store.py` | SQLite 课表/待办/DDL，按 user_id 隔离 |")
    L.append("| 分层记忆 | `memory/` | L0 原文 / L1 事实 / L2 场景 / L3 画像 + 上下文卸载 |")
    L.append("| Guard | `echoguide_guard/` | Prompt 注入检测、限流、审计脱敏 |")
    L.append("")
    L.append("## 运行方式")
    L.append("")
    L.append("```bash")
    L.append("pip install -r requirements.txt -r requirements-dev.txt")
    L.append("python -m uvicorn api.main:app --port 8000   # 后端")
    L.append("cd frontend && npm install && npm run dev    # 前端")
    L.append("```")
    L.append("")
    L.append("配置：复制 `.env.example` 为 `.env` 填写 API Key（ANTHROPIC_API_KEY 等）。")
    L.append("")
    return "\n".join(L)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = collect_files()
    total_lines = sum(count_lines(full) for full, _ in files)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = OUT_DIR / f"echoguide-code-{stamp}.zip"

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("_AI_SUMMARY.md", build_summary(files, total_lines))
        for full, rel in files:
            zf.write(full, rel)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"打包完成: {out_path}")
    print(f"文件数: {len(files)} · 代码行数: {total_lines:,} · 压缩包: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
