# EchoGuide — 西电校园智慧助手

面向西安电子科技大学（西电）师生的校园智能问答助手。基于多 Agent 编排 + 三级记忆 + RAG 知识库架构，覆盖**学业支持、校园生活、校务咨询、IT 支持**四大领域，并内置 **Agent 运行时安全治理**（EchoGuide Guard）与 **LLM-as-Judge 评测**体系。

> 说明：本项目由通用客服系统模板深度场景化改造而来，业务语义（意图体系、Agent 角色、Skills、知识库、评测用例、前端文案）已全部替换为西电校园场景。

📄 技术亮点详解：见 [`docs/技术亮点.md`](docs/技术亮点.md)（原 EchoMind 七大亮点保留/升级核对 + 新增 EchoGuide Guard 安全亮点）

---

## 功能总览

| 领域 | 覆盖内容 | 示例问题 |
|------|----------|----------|
| 🎓 学业支持 | 选课、课表、考试、成绩、绩点、重修、转专业、保研 | "这学期选课什么时候开始？" |
| 🏠 校园生活 | 宿舍、食堂、校车、校园卡、快递、水电、社团 | "南校区食堂几点关门？" |
| 📋 校务咨询 | 校历、请假、奖学金、证明开具、缴费、注册 | "奖学金什么时候评定？" |
| 🖥️ IT 支持 | 教务系统、校园网、VPN、邮箱、统一身份认证 | "教务系统登录不上怎么办？" |

- **多轮对话记忆**：Redis 工作记忆（24h TTL + LLM 自动压缩）+ ChromaDB 情景记忆（跨会话语义检索）+ 用户画像
- **多 Agent 路由**：层次化意图识别（领域 domain × 动作 action，LLM 70% + Embedding 20% + 关键词 10%）→ 领域路由（动作不参与）→ 性能路由 → 降级兜底 → 复合问题并行协作；短句追问自动继承上一轮领域
- **Agentic RAG**：Agent 通过 function calling 工具调用循环自主检索知识库（相关性阈值 + 领域过滤 + overlap 语义分块），回答带引用；工具层以标准 MCP 协议（JSON-RPC 2.0）暴露，任何 MCP 客户端即插即用
- **动态 Skills**：可热加载的业务规范文件（SOP），关键词整词匹配 + 追问历史感知注入 system prompt
- **流式对话**：SSE 逐 token 输出 + 工具调用过程实时可视化（`/chat/stream`）
- **语义缓存**：GPTCache 思路，相似问题直接复用答案（成本趋近于 0）
- **监控闭环**：Prometheus 指标 + Agent 在线表现统计 + 路由惩罚动态调整 + 全链路 Trace（`/traces`）
- **评测体系**：双模型 LLM-as-Judge 四维度评分（生成 ≠ 评判，消除自评偏差）+ 意图/路由/追问评测 + 回归对比
- **安全治理**：EchoGuard 中间件真实接入请求链（认证/注入检测/限流/脱敏审计）+ Sidecar 最小权限策略

---

## 系统架构

```text
┌─────────────┐     ┌──────────────────────────────────────────────────────┐
│  Frontend   │     │                      FastAPI                          │
│ (Vue 3)     │ ──► │  /chat  /chat/stream  /mcp  /search  /knowledge/*     │
│  SSE 流式   │     │  /skills  /eval/run  /traces  /monitor                │
└─────────────┘     └───────────────┬──────────────────────────────────────┘
                                    │
                      ┌─────────────▼──────────────┐
                      │  AgentOrchestrator          │
                      │  ┌───────────────────────┐  │
                      │  │ IntentRecognizer       │  │ 领域×动作 二维意图
                      │  │ 层次化意图 + 追问继承    │  │ LLM+Embedding+Pattern
                      │  └───────────────────────┘  │
                      │  ┌─────┬─────┬─────┬─────┐  │
                      │  │Academic│Campus│Affairs│IT  │ 四领域 Agent
                      │  │  Life │      │  Help  │  │ + function calling
                      │  └─────┴─────┴─────┴─────┘  │ 工具调用循环(Agentic RAG)
                      │  + SkillManager 动态注入 SOP  │
                      └─────────────┬──────────────┘
                                    │
        ┌───────────────┬───────────┼───────────┬─────────────────┐
        ▼               ▼           ▼           ▼                 ▼
   Redis          ChromaDB      ChromaDB     KnowledgeBase  EchoGuide Guard
   工作记忆        情景记忆       用户画像      RAG 向量检索  中间件 + Sidecar
        └───────────────┴───────────┴───────────┴─────────────────┘
     语义缓存(SemanticCache) / Prometheus 监控 / 双模型 LLM-as-Judge / 全链路 Trace
```

## 技术栈

- **后端**：Python 3.12 · FastAPI · Anthropic SDK（兼容 DeepSeek）
- **记忆**：Redis（工作记忆）+ ChromaDB（情景记忆 / 用户画像 / RAG 知识库，内置 all-MiniLM-L6-v2 embedding）
- **前端**：Vue 3 + Vite，支持 Python/Java 双后端调试面板
- **部署**：Docker 多阶段构建（生产镜像 ~500MB）· Docker Compose · Nginx 反代 · Prometheus
- **质量**：pytest（业务逻辑 + EchoGuide Guard 安全测试）· LLM-as-Judge 评测

---

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 填写 ANTHROPIC_API_KEY（Claude 或 DeepSeek 兼容协议）
```

### 2. 本地运行后端

```bash
pip install -r requirements.txt
python -m uvicorn api.main:app --port 8000   # 本地直跑为 8000
```

或启动交互式 CLI：

```bash
python api/main.py --cli
```

> 端口说明：本地直接跑后端监听 **8000**；Docker Compose 部署后，Nginx 统一入口（前端页面 + API 转发）对外暴露 **8088**，API 直连端口为 8100（容器内部 8000）。

### 3. 本地运行前端（可选）

```bash
cd frontend
npm install
npm run dev   # http://localhost:5175
```

### 4. Docker Compose 一键部署

```bash
./build-image.sh
docker compose up -d
```

访问：`http://localhost:8088`（前端页面 + API 统一入口，Swagger 文档 `http://localhost:8088/docs`）

---

## 常用 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | 对话主入口（message / user_id / conv_id），语义缓存优先 |
| POST | `/chat/stream` | 流式对话（SSE）：meta/tool/delta/done 事件序列，工具调用过程可视化 |
| POST | `/mcp` | 标准 MCP 协议端点（JSON-RPC 2.0：initialize/tools/list/tools/call） |
| GET | `/health` | 健康检查 |
| GET | `/monitor` | Agent 实时表现摘要 |
| GET | `/traces` · `/traces/{id}` | 全链路 Trace 列表 / 详情（逐跳耗时） |
| POST | `/search?query=&top_k=` | RAG 检索（改写→召回→重排） |
| POST | `/knowledge/add` | 批量导入知识文档 |
| POST | `/knowledge/upload` | 上传文件导入（txt/md/json） |
| GET | `/skills` · POST `/skills/reload` | Skills 查看 / 热加载 |
| POST | `/eval/run` | 运行评测（意图 + 多轮对话 + 路由/追问） |

## Skills 热加载

业务规范存放于 `skills/`（四个领域 SOP），修改后无需重启：

```bash
curl -X POST http://localhost:8088/skills/reload   # Docker 部署统一入口；本地直跑为 8000
```

## 测试

```bash
python -m pytest tests/ -q
```

覆盖：Skill 解析/匹配/注入、多 Agent 路由/降级/并行协作、二维意图投票与追问继承、子串误命中回归、MCP 协议、EchoGuide Guard 安全策略/中间件/静态扫描、SSE 流式链路、Trace。

## 项目结构

```text
api/              FastAPI 入口、路由、CLI
agents/           多 Agent 编排（四领域 Agent + 工具调用循环 + 路由）
core/             二维意图识别、Skill 加载器、领域词表、轻量 Trace
memory/           三级记忆（工作 / 情景 / 用户画像）
mcp/              RAG 知识库、工具框架、语义缓存、MCP 协议层
evaluation/       双模型 LLM-as-Judge 评测框架（意图/对话/路由）
echoguide_guard/  Agent 运行时安全（Sidecar + 真实接入中间件）
skills/           四领域业务 SOP（可热加载）
frontend/         Vue 3 调试面板（SSE 流式渲染）
config/           Prometheus 等运行配置
data/             demo 知识数据 / ChromaDB 持久化
docs/             文档与历史归档
```
