# EchoGuide · 西电校园智慧助手

EchoGuide 是面向西安电子科技大学学生的校园 Agent，也是一个可观测、可评测的 Agent Runtime。它不把所有请求都送进同一条昂贵链路：课表、校车、办事清单和故障诊断走 **DeepSeek V4 Flash 快速路径**；政策检索、低置信度问题和跨领域依赖任务进入 **DeepSeek V4 Pro 深度路径**。

项目保留多 Agent、三级记忆、Agentic RAG、动态 Skills、MCP、语义缓存、Guard、Trace、Monitor 与 LLM-as-Judge，但每项技术都有明确触发条件和可复现实测，而不是停留在架构图中。

## 三类核心任务

| 任务 | 示例 | 系统行为 |
|---|---|---|
| 个人校园状态 | “我今天有什么课？”“还有哪些 DDL？” | 读取登录用户的课表、待办与考试数据 |
| 确定性校园服务 | “算一下加权成绩”“校园卡怎么补办”“校园网认证后打不开网页” | 调用领域专属工具，返回可测试的结构化结果 |
| 复杂校园任务 | “明天下午有空就安排我去补办校园卡，并记个待办” | Planner 生成依赖 DAG，多个 Agent 分波执行并合成结果 |

## 真实网页实测

以下截图由 Playwright 访问真实网页、登录 Demo 用户、导入动态课表、调用真实 DeepSeek 模型后自动生成；使用 `?debug=1` 展开 Profile、分类阶段、工具、DAG、Token 和 Trace ID。

### Fast · 个人课表

![Fast 个人课表实测](assets/readme/01-fast-personal.png)

### 领域专属工具

![Affairs 专属工具实测](assets/readme/02-specialized-tools.png)

### Deep · Agentic RAG

![Deep RAG 实测](assets/readme/03-deep-rag.png)

### 多 Agent 依赖 DAG

![多 Agent DAG 实测](assets/readme/04-multi-agent-dag.png)

### 多轮记忆与 Guard 拒绝

![多轮记忆与 Prompt 注入拦截实测](assets/readme/05-memory-and-guard.png)

## Fast / Deep 双路径

```mermaid
flowchart LR
    U["学生请求"] --> G["级联意图识别"]
    G -->|"Pattern ≥ 0.90"| C["复杂度闸门"]
    G -->|"Embedding ≥ 0.74 且 margin ≥ 0.08"| C
    G -->|"短追问"| H["历史领域继承"] --> C
    G -->|"低置信度"| L["LLM 分类"] --> C
    C -->|"单领域 / 确定性工具"| F["V4 Flash · Fast"]
    C -->|"RAG / 低置信度"| D["V4 Pro · Deep"]
    C -->|"多领域依赖"| P["Planner → DAG → Executor"]
    P --> D
    F --> T["Tools / MCP"]
    D --> T
    T --> R["回答 + Execution Meta + Trace"]
```

| Profile | 模型 | 思考模式 | 输出预算 | RAG 策略 | 典型任务 |
|---|---|---|---:|---|---|
| Fast | `deepseek-v4-flash` | 关闭 | 768 | Top-K 3，不改写、不重排 | 课表、天气、校车、确定性工具 |
| Deep | `deepseek-v4-pro` | 开启，effort=high | 1536 | Top-K 5，查询改写 + LLM 重排 | 政策问答、复杂请求、多 Agent |

Monitor 按 `academic.fast`、`academic.deep` 等真实 Profile 统计成功率、延迟和在途请求。Fast 执行失败时升级到 Deep；两种 Profile 可以分别配置 API Key、模型和端点。

## 多 Agent 的合理触发边界

简单问题始终走单 Agent。只有两类请求进入协作：

- `parallel`：显式包含“同时、还要、并且、另外、顺便”等复合语义，并命中两个以上领域。
- `dependent`：命中必须使用前序结果的任务规则。

依赖任务示例：

```mermaid
flowchart LR
    S["PersonalAgent<br/>query_schedule"] --> A["PersonalAgent<br/>add_todo"]
    P["AffairsAgent<br/>query_affairs_process"] --> A
    A --> Y["Deep Synthesizer"]
```

Executor 每次最多运行三个 Agent、六个任务；依赖缺失或出现循环时直接失败并记录计划错误，不会绕过 DAG 执行。

## 五个 Agent 与真实能力差异

| Agent | 专属能力 | 工具权限 |
|---|---|---|
| AcademicAgent | 学业政策 RAG、加权学分成绩计算 | `knowledge_search`、`calculate_weighted_score` |
| AffairsAgent | 版本化办事材料、步骤、部门与来源 | `knowledge_search`、`query_affairs_process` |
| ITHelpAgent | 校园网、VPN、统一认证、教务系统诊断树 | `knowledge_search`、`diagnose_it_issue` |
| CampusLifeAgent | 校车、楼宇、场馆、图书馆、天气 | `knowledge_search`、`query_campus_info`、`get_weather` |
| PersonalAgent | 登录用户课表、待办、DDL 与考试 | `query_schedule`、`query_todo`、`add_todo`、`complete_todo`、`query_ddl` |

工具在模型可见范围和执行层分别校验权限。个人工具只接受服务端签名 Cookie 中的身份，客户端提交的 `user_id` 不作为可信身份。

## 真实 Benchmark

Benchmark 使用 12 个版本化场景，覆盖五个领域、追问继承、Fast/Deep 路由、专属工具、RAG、多 Agent DAG 和 Guard。默认每个场景运行三次，并与 Always-LLM + Always-Deep 基线比较。

<!-- BENCHMARK:START -->
> 实测时间：2026-08-09 16:56:50 +0800 · Commit `81affb8` · 每场景重复 3 次

| 指标 | 自适应链路 | Always-LLM + Always-Deep 基线 |
|---|---:|---:|
| 用例通过率 | 100.0% | 63.6% |
| 领域准确率 | 100.0% | 81.8% |
| 领域 Macro-F1 | 100.0% | 73.3% |
| LLM 分类调用率 | 9.1% | 100.0% |
| Profile 路由准确率 | 100.0% | 81.8% |
| 复杂度 Precision / Recall | 100.0% / 100.0% | 100.0% / 100.0% |
| 专属工具成功率 | 100.0% | 66.7% |
| DAG 任务成功率 | 100.0% | 100.0% |
| RAG HitRate@5 / Recall@5 / MRR | 100.0% / 100.0% / 1.00 | — |
| 引用正确率 | 100.0% | 100.0% |
| P50 延迟 | 4641 ms | 6700 ms |
| P95 延迟 | 55040 ms | 47198 ms |
| 输入 / 输出 Token | 84501 / 30436 | 81476 / 28917 |

> 消融：专属工具成功率 100.0%，改用通用 RAG 后为 0.0%；依赖 DAG 成功率 100.0%，强制单 Agent 后为 0.0%。
<!-- BENCHMARK:END -->

完整机器可读结果保存在 [`assets/readme/demo-metrics.json`](assets/readme/demo-metrics.json)。报告记录时间、Git commit、模型、逐场景检查和失败信息，不隐藏不利结果。

## Agent 工程闭环

- **三级记忆**：Redis 工作记忆解决当前对话，ChromaDB 情景记忆恢复跨会话信息，用户画像仅在检测到明确背景/偏好信号时更新。
- **Agentic RAG**：Deep Profile 自主调用知识检索，执行查询改写、并行召回、去重与重排；回答携带来源证据。
- **动态 Skills**：五类校园 SOP 可热加载，按 Agent、关键词和对话历史注入 Prompt。
- **双层语义缓存**：Global 只缓存公共回答；User 按用户和上下文指纹隔离；个人数据领域不缓存。
- **EchoGuide Guard**：Prompt 注入检测、输入限制、用户/IP 限流、审计脱敏与失败关闭。
- **可观测性**：SSE 展示工具过程；`execution` 返回安全执行摘要；Trace 记录逐跳耗时；Prometheus 采集真实成功率/延迟指标，`config/alerts/` 提供告警规则，Monitor 反馈到 Profile 路由。
- **评测**：意图 Accuracy/Macro-F1、RAG HitRate/Recall/MRR、引用正确率、回答忠实性、回归基线和双模型 Judge。
- **MCP**：`POST /mcp` 提供 Streamable HTTP tools 子集。浏览器与 MCP 均使用签名登录 Cookie；未登录客户端只能调用公开工具。

## 快速启动

### 1. 配置

```powershell
Copy-Item .env.example .env
# 编辑 .env，填写 ANTHROPIC_API_KEY
```

生产环境（`APP_ENV=production`）**必须设置 `JWT_SECRET_KEY`**（会话签名密钥），
缺失或保持占位值时服务将拒绝启动（fail-closed）。首次启动可用
`ECHOGUIDE_ADMIN_PASSWORD` 播种管理员账号（仅对新建数据库生效）。

默认 DeepSeek 配置：

```dotenv
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_MODEL=deepseek-v4-flash
ECHOGUIDE_FAST_MODEL=deepseek-v4-flash
ECHOGUIDE_DEEP_MODEL=deepseek-v4-pro
```

### 2. 本地运行

```powershell
pip install -r requirements.txt -r requirements-dev.txt
python -m uvicorn api.main:app --port 8000
```

另一个终端：

```powershell
Set-Location frontend
npm install
npm run dev
```

Vite 开发代理默认转发到 `http://localhost:8000`（与上方 uvicorn 端口一致）；
Docker 部署时后端对外端口为 8100，可用 `VITE_PYTHON_API_URL=http://localhost:8100` 覆盖。

访问 `http://localhost:5175`；技术演示模式为 `http://localhost:5175/?debug=1`。

### 3. Docker Compose

```powershell
docker compose up -d --build
```

统一入口为 `http://localhost:8088`，API 文档为 `http://localhost:8088/api/docs`。

## 复现 Demo、指标与截图

本地 Benchmark 策略覆盖默认关闭。仅在演示环境设置：

```dotenv
ECHOGUIDE_BENCHMARK_ENABLED=1
```

运行 Smoke：

```powershell
python -m evaluation.demo_benchmark --base-url http://localhost:8000 --smoke
```

运行完整三轮真实 Benchmark 并更新 README：

```powershell
python -m evaluation.demo_benchmark --base-url http://localhost:8000 --repeat 3 --update-readme
```

安装浏览器并生成五张真实网页截图：

```powershell
Set-Location frontend
npx playwright install chromium
$env:ECHOGUIDE_DEMO_URL='http://localhost:8088'
npm run demo:capture
```

脚本默认驱动系统 Microsoft Edge；可用 `ECHOGUIDE_PLAYWRIGHT_CHANNEL=chrome` 切换到 Chrome。本地 Vite 模式可额外设置 `ECHOGUIDE_API_URL=http://127.0.0.1:8100`。

截图脚本会自动登录专用 `echoguide_demo` 用户、替换该用户课表并实测 SSE；不会修改其他用户数据。回答失败、执行路径不符或截图缺失时命令返回非零。

## API 摘要

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/chat`、`/chat/stream` | 对话与 SSE；附加向后兼容的 `execution` 字段 |
| POST/GET | `/personal/schedule/*`、`/personal/todo/*` | 登录用户课表、待办、DDL |
| POST | `/mcp` | MCP Streamable HTTP tools 子集 |
| POST | `/search` | RAG 改写、召回、重排演示 |
| POST/GET | `/knowledge/*` | 管理员知识导入与统计 |
| GET | `/monitor`、`/metrics`、`/traces` | 监控、Prometheus 和 Trace |
| POST | `/eval/run` | 管理员评测入口 |

`execution` 只包含路径、Profile、分类阶段、Agent、工具名、任务状态、模型、Token 和 Trace ID；不会返回思维链、完整 Prompt、个人上下文或敏感工具参数。

## 测试

```powershell
python -m pytest tests -q
Set-Location frontend
npm run build
npm audit --omit=dev --audit-level=high
```

CI 运行全部离线回归、前端构建、依赖审计和 Docker Compose 配置检查。真实 DeepSeek Benchmark 与截图任务由开发者手动触发，避免 CI 消耗密钥。

## 项目结构

```text
agents/        Fast/Deep 五领域 Agent、复杂度闸门、Planner/DAG/Executor/Synthesizer
api/           FastAPI、认证、SSE、execution 元数据、MCP 与管理接口
core/          级联意图识别、领域词表、Skills 与 Trace
tools/         个人、校园、Academic、Affairs、IT 确定性工具
data/public/   版本化校园公开信息、办事流程和 IT 诊断树
memory/        工作记忆、情景记忆与用户画像
mcp/           工具管理器、Agentic RAG、协议和语义缓存
config/        Prometheus 抓取配置与告警规则（config/alerts/）
evaluation/    离线评测与真实 Demo Benchmark
frontend/      Vue 3 学生界面、debug 执行详情和 Playwright 截图
assets/readme/ README 截图与真实 Benchmark 结果
```

> 仓库内结构化校园信息包含来源、更新时间和适用范围；无法确认的内容会明确标为演示级数据。实际办理、班次和开放时间应以学校最新官方通知为准。
