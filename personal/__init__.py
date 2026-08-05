"""
个人数据中心 —— 让学生成为「被服务的对象」。

承载三类用户个人数据（按 user_id 隔离）：
  - 课程表（schedule）：ICS / JSON 导入，支持"今天/明天/周几有什么课"查询与地点反查
  - 待办（todo）：作业、杂事清单
  - 考试与 DDL（kind=ddl/exam 的 todo）：倒计时提醒

配套模块：
  - store.py        SQLite 存储层（stdlib sqlite3，零第三方依赖）
  - ics_parser.py   轻量 ICS 课程表解析（不依赖 icalendar 库）
  - time_context.py 时间上下文（当前日期/星期/第几节/第几周），注入 Agent system prompt
  - service.py      查询服务（日期表达式解析、课程查询、DDL 倒计时、提醒汇总）
"""
