# 西电校园助手（EchoGuide）Skills 文档

EchoGuide 启动时会从 `ECHOGUIDE_SKILLS_DIR` 读取 Skills，并在匹配用户请求时注入到对应 Agent 的 system prompt。Skills 适合维护校园各业务领域的答复规范、办事流程 SOP、排障指引、升级（转人工/转部门）规则和禁止事项。

当前内置四类 Skills，对应四个领域 Agent：

```text
skills/academic/SKILL.md     # 学业支持：选课、课表、考试、成绩、绩点、重修、保研、转专业
skills/campus_life/SKILL.md  # 校园生活：宿舍、食堂、校车、校园卡、快递、水电、社团
skills/affairs/SKILL.md      # 校务咨询：校历、请假、奖学金、助学金、证明开具、缴费、注册
skills/it_help/SKILL.md      # IT 助手：教务系统、校园网、VPN、邮箱、统一身份认证排障
```

## Skill 文件格式

推荐每个 Skill 使用独立目录，并将主文件命名为 `SKILL.md`：

```text
skills/<skill_name>/SKILL.md
```

文件顶部使用简单 front matter：

```markdown
---
name: 学业咨询规范
description: 适用于 AcademicAgent 的选课、成绩等学业问题答复规范
keywords: 选课,课表,考试,成绩,绩点,学分,重修,保研
agents: academic
enabled: true
---
```

字段说明：

- `name`：Skill 展示名称，会出现在注入给模型的 prompt 中。
- `description`：简短说明，方便 `/skills` 接口排查。
- `keywords`：触发关键词，用户消息命中后才注入；多个关键词用英文逗号或中文逗号分隔均可。
- `agents`：适用 Agent，可填 `academic`、`campus_life`、`affairs`、`it_help`，多个值用逗号分隔。
- `enabled`：是否启用，支持 `true/false`。

## 编写要求

- 重要规则放在文档前半部分，因为过长内容会按 prompt 预算截断。
- 一类 Skill 只描述一类职责，不要把学业、生活、校务、IT 规则混在一个文件里。
- 必须包含"角色定位""处理流程""禁止事项""示例表达"等稳定章节。
- 对密码、验证码、完整证件号、银行卡号等敏感信息必须写明禁止收集。
- 对无法保证的事项（具体分数、截止日期、金额）使用保守措辞，例如"通常""以最新通知为准"。
- 对需要联系辅导员、教务老师、网络中心、学生处等处理的场景要明确写出引导条件。

## 匹配规则（v2 升级）

- **整词匹配**：ASCII 关键词（如 `vpn`）强制词边界匹配，`api` 不会命中 `capital`；中文关键词必须 ≥2 字，**禁止单字关键词**（如旧版"餐"会命中"餐补"等无关场景，请改为"餐厅/早餐/午餐/晚餐"等多字词组）。
- **追问继承**：当前消息未命中关键词时，会自动回溯最近 2 轮用户消息匹配 —— 用户追问"那几点开门呢？"仍能命中上一轮"南校区食堂几点关门？"的 `食堂` 关键词并注入对应 SOP。
- 关键词与 Agent 类型**双条件**同时满足才注入；`agents` 留空表示全局 Skill。

## 热加载

修改 Skill 文件后，不需要重启服务，调用：

```bash
curl -X POST http://localhost:8088/skills/reload   # Docker 部署统一入口（Nginx）；本地直跑后端为 8000
```

查看加载结果和解析错误：

```bash
curl http://localhost:8088/skills
```
```
