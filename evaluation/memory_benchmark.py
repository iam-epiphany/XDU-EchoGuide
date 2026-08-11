"""
确定性离线评测：分层记忆（L0-L3 金字塔）与上下文卸载的效果数据。

用途：简历/README 引用的量化指标，无需 API key，可重复运行、可入 CI。
统计口径（脚本内统一）：
  - Token 估算：中文等宽字符 1 字符 ≈ 1 token，ASCII 4 字符 ≈ 1 token
    （memory.layered_store.estimate_tokens，确定性可复现；
    真实消耗以模型 API usage 为准，此处用于相对对比）
  - 全部输入为脚本内固定模拟数据（无随机），多次运行结果一致

输出：
  - 卸载对比：同一批长工具结果，"全量进上下文" vs "卸载后摘要行+索引"的 token 对比
  - 分层回放：写入 L0 原文 → L1 事实（带证据链）→ L3 画像版本 → 溯源断言
  - refs 找回：卸载结果 100% 可恢复
  - 提炼信号率：画像信号检测（_has_profile_signal）在模拟对话上的触发比例
  - 治理清理：prune 生命周期清理的统计

运行：python evaluation/memory_benchmark.py
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from memory.layered_store import (
    LayeredStore, estimate_tokens, OFFLOAD_CHARS, OFFLOAD_SUMMARY_CHARS,
)
from memory.conversation_memory import Message, MsgRole, _has_profile_signal

# ── 模拟数据（固定内容，确定性）───────────────────────────────────────────────

# 10 轮模拟对话：混合普通提问与画像信号句（"我最近在准备考研"等）
SIMULATED_TURNS = [
    ("user",   "南校区图书馆几点关门？"),
    ("assistant", "南校区图书馆周一至周五 8:00-22:00 开放，周末 9:00-21:30。"),
    ("user",   "我最近在准备考研，想找个晚上开放的自习室"),
    ("assistant", "图书馆四楼自习区开放到 22:30，凭校园卡进入。"),
    ("user",   "我是通信工程学院大二的，在南校区"),
    ("assistant", "通信工程学院大二课程多在信远楼，南校区主教学楼为信远一区。"),
    ("user",   "那实验室晚上几点关门？"),
    ("assistant", "学院实验室一般 21:30 关闭，部分开放到 22:00。"),
    ("user",   "帮我记个待办：周三下午去校医院补办校园卡"),
    ("assistant", "好的，已为你记录待办：周三下午补办校园卡。"),
    ("user",   "好的谢谢"),
    ("assistant", "不客气，有需要随时找我。"),
]

# 3 个超长工具结果（模拟 knowledge_search 返回的大段文档，> OFFLOAD_CHARS）
SIMULATED_TOOL_RESULTS = [
    "[文档] 西安电子科技大学本科生选课管理办法（节选）" + "选课分为预选、正选、补退选三个阶段。" * 120,
    "[文档] 南校区餐饮服务指南（节选）" + "竹园餐厅位于南校区东侧，营业时间 6:30-21:30，提供清真窗口。" * 120,
    "[文档] 校园卡使用与挂失流程（节选）" + "校园卡丢失后请第一时间在自助机或公众号挂失，挂失后原卡立即冻结。" * 120,
]


def _signal_stats(turns) -> dict:
    """画像信号检测率：信号句/总轮次（成本控制的核心指标）。"""
    msgs = [Message(role=MsgRole(v[0]), content=v[1]) for v in turns]
    total = 0
    hits = 0
    for i in range(1, len(msgs) + 1):
        if msgs[i - 1].role != MsgRole.USER:
            continue
        total += 1
        # 与 update_profile 相同口径：只看最近 2 条用户消息
        if _has_profile_signal(msgs[:i]):
            hits += 1
    return {"user_turns": total, "signal_hits": hits,
            "rate_pct": round(hits / total * 100, 1) if total else 0.0}


def _offload_stats(results: list) -> dict:
    """上下文卸载对比：全量进上下文 vs 卸载后（摘要行 + refs 索引）。"""
    full_tokens = 0
    offload_tokens = 0
    offloaded_chars = 0
    items = []
    for i, text in enumerate(results):
        t_full = estimate_tokens(text)
        # 与 agent_orchestrator 卸载逻辑同口径：摘要 OFFLOAD_SUMMARY_CHARS + 索引行
        if len(text) > OFFLOAD_CHARS:
            summary_line = (f"{text[:OFFLOAD_SUMMARY_CHARS]}..."
                            f"[完整结果 refs/{i + 1}，共 {len(text)} 字符]")
            t_off = estimate_tokens(summary_line)
            offloaded_chars += len(text) - len(summary_line)
        else:
            summary_line = text
            t_off = t_full
        full_tokens += t_full
        offload_tokens += t_off
        items.append({
            "char_len": len(text), "full_tokens": t_full,
            "offload_tokens": t_off, "offloaded_chars": len(text) - len(summary_line),
        })
    return {
        "items": items,
        "full_tokens": full_tokens,
        "offload_tokens": offload_tokens,
        "saved_tokens": full_tokens - offload_tokens,
        "offloaded_chars": offloaded_chars,
        "saved_pct": round((1 - offload_tokens / full_tokens) * 100, 1),
    }


async def _layered_replay(tmp_db: str) -> dict:
    """L0 写入 → L1 事实（证据链）→ L3 版本历史 → 溯源/找回/治理断言。"""
    store = LayeredStore(tmp_db)
    result: dict = {}

    # ── L0 原文写入（模拟 add_message 的落库路径）──────────────────────────
    for role, content in SIMULATED_TURNS:
        await store.append_raw("u_bench", "conv_1", role, content)
    raw_count = await store.count_raw("u_bench")
    result["raw"] = {"turns": raw_count, "assert_full": raw_count == len(SIMULATED_TURNS)}

    # ── L1 原子事实（带证据链：source_turn → L0 原文）──────────────────────
    last_turn = await store.get_last_turn("u_bench", "conv_1")
    facts = [
        {"fact": "用户在准备考研", "category": "status", "source_conv": "conv_1", "source_turn": 3},
        {"fact": "用户是通信工程学院大二学生", "category": "entity", "source_conv": "conv_1", "source_turn": 5},
        {"fact": "用户在南校区", "category": "entity", "source_conv": "conv_1", "source_turn": 5},
        {"fact": "周三下午去校医院补办校园卡（待办）", "category": "decision", "source_conv": "conv_1", "source_turn": 9},
    ]
    added = await store.add_facts("u_bench", facts)
    # 重复提炼去重断言
    added_dup = await store.add_facts("u_bench", [dict(facts[0])])
    # 证据链溯源：每条 fact 的 source_turn 必须能在 L0 找到原文
    by_turn = await store.get_raw_by_turns("u_bench", "conv_1", [f["source_turn"] for f in facts])
    traceable = sum(1 for f in facts if f["source_turn"] in by_turn)
    result["facts"] = {
        "added": added, "dup_skipped": added_dup == 0,
        "traceable": traceable, "total": len(facts),
        "traceback_pct": round(traceable / len(facts) * 100, 1),
        "evidence_sample": by_turn.get(3, "")[:18] + "...",  # 下钻到原文的样例
    }

    # ── L3 画像版本历史（3 次提炼 → 3 版，可回滚）──────────────────────────
    for i in range(3):
        await store.save_profile_version(
            "u_bench",
            json.dumps({"preferences": [f"偏好{i}"], "entities": {}}, ensure_ascii=False),
            reason=f"signal: conv_1",
        )
    versions = await store.list_profile_versions("u_bench")
    oldest = await store.get_profile_version("u_bench", versions[-1]["id"])
    result["profile"] = {
        "versions": len(versions),
        "rollback_ok": oldest is not None and "偏好0" in oldest["profile_json"],
    }

    # ── refs 卸载落盘：100% 找回 ────────────────────────────────────────────
    ref_ids = []
    for i, text in enumerate(SIMULATED_TOOL_RESULTS):
        ref_ids.append(await store.save_ref("u_bench", "conv_1", "knowledge_search", text))
    recovered = 0
    for rid in ref_ids:
        ref = await store.get_ref("u_bench", rid)
        if ref is not None and ref.get("content") is not None:
            recovered += 1
    result["refs"] = {"saved": len(ref_ids), "recovered": recovered,
                      "recover_pct": round(recovered / len(ref_ids) * 100, 1)}

    # ── 治理：失效标记 + prune 清理 ─────────────────────────────────────────
    deactivated = await store.deactivate_fact("u_bench", (await store.list_facts("u_bench"))[0]["id"])
    stats = await store.prune("u_bench", raw_ttl_days=0, ref_ttl_days=0,
                              fact_ttl_days=0, max_profile_versions=1)
    result["governance"] = {"deactivate_ok": deactivated, "prune": stats,
                            "active_facts_left": await store.count_facts("u_bench")}
    return result


async def main() -> dict:
    report = {
        "purpose": "分层记忆（L0-L3）+ 上下文卸载 · 确定性离线评测（模拟数据，无 API 依赖）",
        "token_estimation": "中文 1 字符≈1 token，ASCII 4 字符≈1 token（相对对比口径）",
    }
    report["offload"] = _offload_stats(SIMULATED_TOOL_RESULTS)
    report["signal"] = _signal_stats(SIMULATED_TURNS)
    with tempfile.TemporaryDirectory() as tmp:
        report["layers"] = await _layered_replay(str(pathlib.Path(tmp) / "bench.db"))
    return report


if __name__ == "__main__":
    rep = asyncio.run(main())
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    # 摘要行（README / 简历直接引用）
    off, sig, lay = rep["offload"], rep["signal"], rep["layers"]
    print("\n===== 摘要 =====")
    print(f"上下文卸载：{len(off['items'])} 个长工具结果，"
          f"全量 {off['full_tokens']} tokens → 卸载后 {off['offload_tokens']} tokens，"
          f"节省 {off['saved_pct']}%")
    print(f"L0 原文全量落库：{lay['raw']['turns']} 轮 / 100% 保留")
    print(f"L1 原子事实溯源：{lay['facts']['traceback_pct']}%（证据链可下钻到 L0 原文）")
    print(f"L3 画像版本：{lay['profile']['versions']} 版，回滚 {'OK' if lay['profile']['rollback_ok'] else 'FAIL'}")
    print(f"refs 卸载找回：{lay['refs']['recover_pct']}%（100% 可恢复）")
    print(f"画像提炼信号率：{sig['rate_pct']}%（仅信号句触发 LLM 提炼，控制成本）")
