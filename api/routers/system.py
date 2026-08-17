"""系统与支撑路由：/health、/skills、/campus。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api import state
from api.deps import require_admin

router = APIRouter(tags=["系统"])


@router.get("/health")
async def health():
    if state._orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {
        "status": "ok",
        "agents": state._orchestrator.get_stats(),
        "verification": state._orchestrator.verification_stats(),
    }


@router.get("/skills")
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if state._skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return state._skill_manager.summary()


@router.post("/skills/reload")
async def reload_skills(_admin=Depends(require_admin)):
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if state._skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    state._skill_manager.reload()
    if state._orchestrator is not None:
        state._orchestrator.set_skill_manager(state._skill_manager)
    return state._skill_manager.summary()


@router.get("/campus/info")
async def campus_info(category: str = "shuttle", keyword: str = ""):
    """
    查询校园公开信息（结构化数据）。
    category: shuttle（校车下一班，keyword 传方向）/ buildings（楼宇）/ venues（场馆）/ library（图书馆）。
    """
    if state._campus_store is None:
        raise HTTPException(503, "公开信息数据源未初始化")
    return state._campus_store.search(category, keyword)


@router.post("/campus/reload")
async def campus_reload(_admin=Depends(require_admin)):
    """热加载 data/public/*.json（填充真实数据后无需重启）。"""
    if state._campus_store is None:
        raise HTTPException(503, "公开信息数据源未初始化")
    return {"status": state._campus_store.reload()}
