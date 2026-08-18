---
name: 待办管理
description: 查询、新增、完成或恢复用户个人待办、考试和 DDL 记录
keywords: 待办,记一下,提醒,完成待办,删除待办,作业,ddl,考试
enabled: true
---
# Goal
在运行时权限边界内管理当前用户的个人事项，并提供明确回执。

# Procedure
1. 查询用 `query_todo` 或 `query_ddl`；新增前确认 content、kind 和可确定的 due_at。
2. 写入用 `add_todo`，完成/恢复用 `complete_todo`；仅在 Action/Role 允许时执行。
3. 工具结果返回后，复述已执行的事项、日期和编号；失败时如实说明。

# Gotchas / Failure
- “删除”当前没有对应工具时不可假装完成；日期歧义先澄清或明确按何种日期解释。

# Output Contract
查询输出按紧急程度排序；写入输出操作回执与记录标识。
