"""MCPToolManager 检索优化链路测试：use_rewrite 工具经 call() 自动走改写链路。

覆盖：
  1. use_rewrite=True 的工具：call() 自动进入「查询改写→并行召回→去重→重排」链路，
     额外参数（min_score/domain）透传给每个子查询，且无递归
  2. 未开启 use_rewrite 的工具：保持原调用路径
  3. 显式 use_rewrite=False：即使工具开启也强制绕过
  4. LLM 不可用时的降级：改写失败退化为原查询、重排失败返回原顺序

所有测试只测确定性逻辑，LLM 调用全部 mock，不触发真实网络请求。
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from mcp.tool_manager import MCPToolManager, Tool

FAKE_KEY = "sk-test-not-used"


def _search_tool(use_rewrite: bool = True) -> Tool:
    """注册一个带调用记录的假检索工具。"""
    calls = []

    async def handler(params, context):
        calls.append(params)
        return [
            {"title": f"文档{i}", "content": f"内容{i}", "score": 0.9 - i * 0.1}
            for i in range(3)
        ]

    tool = Tool(
        name="fake_search",
        description="假检索工具",
        handler=handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
                "min_score": {"type": "number"},
                "domain": {"type": "string"},
            },
            "required": ["query"],
        },
        use_rewrite=use_rewrite,
    )
    tool.calls = calls  # 测试探针
    return tool


def test_use_rewrite_tool_routes_through_rewrite_chain():
    """use_rewrite=True 的工具：call() 自动走改写链路，额外参数透传、无递归。"""
    tm = MCPToolManager(api_key=FAKE_KEY)
    tool = _search_tool(use_rewrite=True)
    tm.register(tool)

    with patch.object(
        tm, "rewrite_query",
        new=AsyncMock(return_value=["选课流程", "选课分几个阶段", "选课有什么要求"]),
    ), patch.object(
        tm, "_rerank",
        new=AsyncMock(side_effect=lambda q, items, k: items[:k]),
    ):
        result = asyncio.run(tm.call(
            "fake_search",
            {"query": "选课流程", "domain": "academic", "min_score": 0.3, "top_k": 5},
        ))

    assert result.success
    assert result.reranked is True
    # 三个子查询各召回一次（改写生效）
    assert len(tool.calls) == 3
    assert {p["query"] for p in tool.calls} == {"选课流程", "选课分几个阶段", "选课有什么要求"}
    # 额外参数原样透传：领域过滤/相关性阈值在改写链路中不丢失
    for params in tool.calls:
        assert params["domain"] == "academic"
        assert params["min_score"] == 0.3


def test_tool_without_use_rewrite_keeps_plain_path():
    """未开启 use_rewrite 的工具：保持原调用路径，不走改写链路。"""
    tm = MCPToolManager(api_key=FAKE_KEY)
    tool = _search_tool(use_rewrite=False)
    tm.register(tool)

    with patch.object(tm, "rewrite_query", new=AsyncMock()) as rw:
        result = asyncio.run(tm.call("fake_search", {"query": "食堂", "top_k": 3}))

    assert result.success
    assert result.reranked is False
    assert len(tool.calls) == 1
    assert tool.calls[0]["query"] == "食堂"
    rw.assert_not_awaited()  # 未走改写链路


def test_explicit_use_rewrite_false_bypasses_chain():
    """显式 use_rewrite=False：即使工具开启也强制绕过（供子查询防递归等场景）。"""
    tm = MCPToolManager(api_key=FAKE_KEY)
    tool = _search_tool(use_rewrite=True)
    tm.register(tool)

    with patch.object(tm, "rewrite_query", new=AsyncMock()) as rw:
        result = asyncio.run(tm.call("fake_search", {"query": "食堂"}, use_rewrite=False))

    assert result.success
    assert result.reranked is False
    assert len(tool.calls) == 1
    rw.assert_not_awaited()


def test_rewrite_failure_degrades_to_original_query():
    """LLM 改写失败：退化为原始查询，链路不中断。"""
    tm = MCPToolManager(api_key=FAKE_KEY)
    tool = _search_tool(use_rewrite=True)
    tm.register(tool)

    with patch.object(
        tm._client.messages, "create",
        new=AsyncMock(side_effect=RuntimeError("LLM 不可用")),
    ), patch.object(
        tm, "_rerank", new=AsyncMock(side_effect=lambda q, items, k: items[:k]),
    ):
        result = asyncio.run(tm.call("fake_search", {"query": "食堂", "top_k": 3}))

    assert result.success
    assert result.reranked is True  # 链路仍然完整返回
    assert len(tool.calls) == 1  # 改写失败退化为单查询，且无递归
    assert tool.calls[0]["query"] == "食堂"


def test_rewrite_query_llm_failure_returns_original_query():
    """rewrite_query 自身：LLM 异常时返回 [原始查询]（内部降级）。"""
    tm = MCPToolManager(api_key=FAKE_KEY)
    with patch.object(
        tm._client.messages, "create",
        new=AsyncMock(side_effect=RuntimeError("LLM 不可用")),
    ):
        out = asyncio.run(tm.rewrite_query("食堂几点关门"))

    assert out == ["食堂几点关门"]


def test_rerank_llm_failure_returns_original_order():
    """_rerank 自身：LLM 异常时返回原始顺序 Top-K（内部降级）。"""
    tm = MCPToolManager(api_key=FAKE_KEY)
    items = [{"title": "a"}, {"title": "b"}, {"title": "c"}]
    with patch.object(
        tm._client.messages, "create",
        new=AsyncMock(side_effect=RuntimeError("LLM 不可用")),
    ):
        out = asyncio.run(tm._rerank("q", items, 2))

    assert out == [{"title": "a"}, {"title": "b"}]  # 原顺序取 Top-2
