# EchoGuide Skills

EchoGuide 从 `ECHOGUIDE_SKILLS_DIR` 发现 `skills/<skill-id>/SKILL.md`。目录（id、名称、描述、关键词）常驻提示；模型按需调用唯一只读工具 `load_skill(skill_name)` 获取完整 SOP。加载 Skill 不会授予工具、写入或角色权限。

内置 12 个任务能力包：

```text
course-planning                 exam-grade-guidance
academic-policy-consulting      campus-card-service
campus-facility-guide           leave-and-certificate
scholarship-aid                 campus-network-troubleshooting
account-auth-troubleshooting    schedule-query
todo-management                 time-planning
```

## 文件格式

```markdown
---
name: 校园网络排障
description: 分层诊断校园网、VPN 和认证故障
keywords: 校园网,vpn,登录不上
enabled: true
---

# Goal
...
```

所有 Skill 平级、默认可发现；Domain 只提供语境，不会过滤 Skill 或 Tool。关键词只作命中提示，最终由模型结合对话决定是否加载。关键词对 ASCII 采用整词匹配，对中文要求至少两个字；当前消息无命中时会回溯最近两轮用户消息。

将 Skill 写成任务 SOP / Runbook：`Goal`、`Procedure`、`Decision Branches`（需要时）、`Tool Usage`、`Gotchas`、`Failure / Escalation`、`Output Contract`。不要在 Skill 中重复 Runtime、Action、Role 的权限规则，也不得借由 Skill 声明写权限。

职责边界：`Domain` 管语境，`Skill` 管方法，`Action` 管允许做什么，`Tool` 管执行，`Runtime` 管生命周期，`Verifier` 管结果。现有 Trace 会记录 `skills_prompted` 与 `skills_loaded`，可通过 `/traces` 和 Eval 排查技能提示、实际加载与工具协作。

修改文件后可通过 `POST /skills/reload` 热加载；`GET /skills` 查看发现结果和解析错误。
