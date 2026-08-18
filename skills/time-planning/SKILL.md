---
name: 时间规划
description: 组合个人课表、待办和 DDL，生成当天或近期的可执行时间计划
keywords: 时间规划,安排一下,今天安排,本周计划,最近要做,怎么安排,优先级
enabled: true
---
# Goal
用真实个人数据建立时间计划；先查询，再建议，写入必须经过现有 Action/Executor 权限。

# Procedure
1. 确认规划范围（今天/本周/截止日期）与固定约束。
2. 先并用 `query_schedule`、`query_todo` 和 `query_ddl` 获取课程、未完成事项和临近 DDL。
3. 按不可移动课程、到期风险、预计工作块排序；标出冲突、空档和需要用户决定的取舍。
4. 仅在用户明确请求且运行时允许时，调用 `add_todo` 记录新事项；规划本身不绕过 QA/Executor 或写权限。

# Gotchas / Failure
- 未登录、未导入或工具不可用时，说明缺失数据，不编造日程；不要把建议当作已写入记录。

# Output Contract
输出数据依据、按时间的计划、风险/冲突和可选的下一步操作。
