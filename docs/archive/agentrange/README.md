# AgentRange 接入 EchoGuard

该集成使用 Compose overlay 将 OpsPilot 的 OpenAI 兼容请求和全部 MCP JSON-RPC 请求路由到 EchoGuard。靶场源码以只读卷挂载，供静态资产扫描使用。

## 启动

先将赛题 ZIP 解压到独立目录，并进入包含 AgentRange `docker-compose.yml` 的目录。不要把靶场源码复制进 EchoMind Git 仓库。

在 PowerShell 中执行：

```powershell
$env:ECHOGUARD_ROOT = (Resolve-Path 'D:\Agent-Project\EchoMind\EchoMind').Path
$overlay = Join-Path $env:ECHOGUARD_ROOT 'integrations\agentrange\docker-compose.echoguard.yml'
docker compose -f .\docker-compose.yml -f $overlay up -d --build
```

健康检查与静态扫描结果：

```powershell
Invoke-RestMethod http://localhost:8200/health
Invoke-RestMethod -Method Post http://localhost:8200/api/v1/assets/scan
Invoke-RestMethod http://localhost:8200/api/v1/assets/latest
```

## 通过代理回放

回放控制面仍直接访问 `llm-stub`，EchoGuard 不读取也不代理 `/admin/trajectories`。业务流量通过 Agent/Langflow 入口代理：

```powershell
$env:OPSPILOT_BASE = 'http://localhost:8200/agent'
$env:LANGFLOW_BASE = 'http://localhost:8200/langflow'
$env:LLM_STUB_BASE = 'http://localhost:18000'
python .\scenario-runner\runner.py
```

如需使用其他宿主机端口，可在启动前设置 `$env:AGENTRANGE_LLM_PORT`，并让 `LLM_STUB_BASE` 使用同一端口。

查看决策与链路：

```powershell
Invoke-RestMethod 'http://localhost:8200/api/v1/traces?limit=20'
Invoke-RestMethod 'http://localhost:8200/api/v1/traces/<trace-id>'
```

Prometheus 指标位于 `http://localhost:8200/metrics`，Swagger 位于 `http://localhost:8200/docs`。

## 当前验证基线

使用靶场默认 `seed=1337` 生成 5200 条事件，并回放全部 80 条恶意事件以及四类正常角色各 5 条样本：

- 20/20 正常样本通过，未产生阻断审计。
- 24/24 Cat-1 样本在 Langflow 执行前返回 403。
- 56/56 Cat-2/M-1 样本在 MCP 链路命中阻断规则。
- 真实 ZIP 静态扫描发现 36 个资产、21 条依赖边、11 个高危或严重问题。

这只是集成回归基线，不等同于最终比赛指标；提交前仍需回放全部 5120 条正常事件，计算误报率和策略延迟分位数。

## 当前边界

- 在线路径只使用确定性策略，不调用 LLM；目标是将单次策略开销控制在毫秒级。
- Trace/Instance ID 仅作为不透明关联键，策略不会解析 `C2-*`、`B-*` 等前缀。
- Cat-2 在 MCP 执行前阻断；Cat-1 通过 Langflow 入口代理阻断。
- `notes-sync` 的容器启动外联不经过 Agent/LLM/MCP 请求链，后续需要 egress 或主机传感器补齐。
- 当前代理模式通过环境变量重路由；如果业务服务仍可被直接访问，存在绕过代理的可能，生产化需要网络策略强制流量经过 Sidecar。
