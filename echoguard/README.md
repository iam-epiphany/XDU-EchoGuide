# EchoGuard

EchoGuard 是 EchoGuide（西电校园智慧助手）仓库中的独立 Agent 运行时安全 Sidecar，为校园助手的多 Agent 调用链提供「最小权限」安全治理，验证以下闭环：

```text
资产扫描 -> 请求来源绑定 -> 模型计划观测 -> MCP 执行前策略
        -> 工具结果污染/敏感标记 -> 阻断审计 -> Trace 还原
```

核心保护面：

- Agent 入口：绑定 JWT 身份、原始任务、Skill 与 Trace。
- OpenAI 兼容代理：记录模型工具计划，识别 Skill/上下文中的注入指令。
- MCP 网关：按校园角色（学生/辅导员/IT 人员/管理员）、学院租户、Skill 能力、路径、命令与数据标签执行 `allow / ask / block`。
- Langflow 入口：阻断危险代码校验载荷和路径穿越上传。
- 静态扫描：识别 Agent、模型、Skills、MCP、网络与高风险配置/供应链组件。

校园场景预设：MCP 上游为教务系统（jwxt）、学籍库（student-db）、校园网（network）、校历（calendar）、通知公告（notice）、校园知识库（knowledge）等；学籍类工具结果自动标记为 PII 并脱敏。

运行时只保存脱敏参数、摘要和哈希，不保存 `.env`、Token 或完整学籍记录。

本地回归：

```powershell
python -m pip install -r .\echoguard\requirements.txt -r .\requirements-dev.txt
python -m pytest -q
```
