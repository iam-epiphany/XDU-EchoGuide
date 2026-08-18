---
name: 校园设施导览
description: 查询校车、图书馆、场馆和楼宇位置、开放时间或下一班信息
keywords: 校车,班车,图书馆,场馆,楼宇,开放时间,几点关门,南校区,北校区
enabled: true
---
# Goal
以结构化校园公开信息回答位置、开放和出行问题。

# Procedure
1. 识别对象、校区、方向和日期；追问沿用上一轮对象。
2. 调用 `query_campus_info`，选用 shuttle、buildings、venues 或 library 分类。
3. 返回工具数据中的位置、下一班或开放状态；若数据源不可用，给出官方公告核实路径。

# Gotchas / Failure
- 不凭记忆编造班次或开放时间；节假日、临时闭馆和维护以现场公告为准。

# Output Contract
给出对象、校区、当前可用信息、时间敏感提示。
