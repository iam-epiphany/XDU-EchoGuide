# EchoGuide — 西电校园智慧助手

面向西安电子科技大学（西电）师生的校园智能问答助手。基于多 Agent 编排 + 三级记忆 + RAG 知识库架构，覆盖**学业支持、校园生活、校务咨询、IT 支持**四大领域，并内置 **Agent 运行时安全治理**（EchoGuard）与 **LLM-as-Judge 评测**体系。

> 说明：本项目由通用客服系统模板深度场景化改造而来，业务语义（意图体系、Agent 角色、Skills、知识库、评测用例、前端文案）已全部替换为西电校园场景。

📄 技术亮点详解：见 [`docs/技术亮点.md`](docs/技术亮点.md)（原 EchoMind 七大亮点保留/升级核对 + 新增 EchoGuard 安全亮点）

---

## 功能总览

| 领域 | 覆盖内容 | 示例问题 |
|------|----------|----------|
| 🎓 学业支持 | 选课、课表、考试、成绩、绩点、重修、转专业、保研 | "这学期选课什么时候开始？" |
| 🏠 校园生活 | 宿舍、食堂、校车、校园卡、快递、水电、社团 | "南校区食堂几点关门？" |
| 📋 校务咨询 | 校历、请假、奖学金、证明开具、缴费、注册 | "奖学金什么时候评定？" |
| 🖥️ IT 支持 | 教务系统、校园网、VPN、邮箱、统一身份认证 | "教务系统登录不上怎么办？" |

- **多轮对话记忆**：Redis 工作记忆（24h TTL + LLM 自动压缩）+ ChromaDB 情景记忆（跨会话语义检索）+ 用户画像
- **多 Agent 路由**：三路融合意图识别（LLM 70% + Embedding 20% + 关键词 10%）→ 三层路由决策（意图映射 / 性能路由 / 降级兜底）→ 复合问题并行协作
- **RAG 知识库**：ChromaDB 向量检索，查询改写 → 并行召回 → LLM 重排链路
- **动态 Skills**：可热加载的业务规范文件（SOP），按关键词 + Agent 类型注入 system prompt
- **监控闭环**：Prometheus 指标 + Agent 在线表现统计 + 路由惩罚动态调整
- **评测体系**：LLM-as-Judge 四维度评分（相关性/准确性/完整性/可执行性）+ 意图识别评测 + 回归对比
- **安全治理**：EchoGuard Sidecar 对 Agent 调用链做最小权限策略、敏感数据脱敏、注入检测、审计追踪

---

## 系统架构

```text
┌─────────────┐     ┌──────────────────────────────────────────────────────┐
│  Frontend   │     │                      FastAPI                          │
│ (Vue 3)     │ ──► │  /chat  /search  /knowledge/*  /skills  /eval/run     │
└─────────────┘     └───────────────┬──────────────────────────────────────┘
                                    │
                      ┌─────────────▼──────────────┐
                      │  AgentOrchestrator          │
                      │  ┌───────────────────────┐  │
                      │  │ IntentRecognizer       │  │  LLM + Embedding + Pattern
                      │  │ 三路融合意图识别         │  │
                      │  └───────────────────────┘  │
                      │  ┌─────┬─────┬─────┬─────┐  │
                      │  │Academic│Campus│Affairs│IT  │ 四领域 Agent
                      │  │  Life │      │  Help  │  │
                      │  └─────┴─────┴─────┴─────┘  │
                      │  + SkillManager 动态注入 SOP  │
                      └─────────────┬──────────────┘
                                    │
        ┌───────────────┬───────────┼───────────┬─────────────────┐
        ▼               ▼           ▼           ▼                 ▼
   Redis          ChromaDB      ChromaDB     KnowledgeBase    EchoGuard
   工作记忆        情景记忆       用户画像      RAG 向量检索    安全 Sidecar
        └───────────────┴───────────┴───────────┴─────────────────┘
                      Prometheus 监控 / LLM-as-Judge 评测
```

## 技术栈

- **后端**：Python 3.12 · FastAPI · Anthropic SDK（兼容 DeepSeek）
- **记忆**：Redis（工作记忆）+ ChromaDB（情景记忆 / 用户画像 / RAG 知识库，内置 all-MiniLM-L6-v2 embedding）
- **前端**：Vue 3 + Vite，支持 Python/Java 双后端调试面板
- **部署**：Docker 多阶段构建（生产镜像 ~500MB）· Docker Compose · Nginx 反代 · Prometheus
- **质量**：pytest（业务逻辑 + EchoGuard 安全测试）· LLM-as-Judge 评测

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
python -m uvicorn api.main:app --port 8000
```

或启动交互式 CLI：

```bash
python api/main.py --cli
```

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

访问：API 文档 `http://localhost:8000/docs` · 前端 `http://localhost:5175`

---

## 常用 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | 对话主入口（message / user_id / conv_id） |
| GET | `/health` | 健康检查 |
| GET | `/monitor` | Agent 实时表现摘要 |
| POST | `/search?query=&top_k=` | RAG 检索（改写→召回→重排） |
| POST | `/knowledge/add` | 批量导入知识文档 |
| POST | `/knowledge/upload` | 上传文件导入（txt/md/json） |
| GET | `/skills` · POST `/skills/reload` | Skills 查看 / 热加载 |
| POST | `/eval/run` | 运行评测（意图 + 多轮对话） |

## Skills 热加载

业务规范存放于 `skills/`（四个领域 SOP），修改后无需重启：

```bash
curl -X POST http://localhost:8000/skills/reload
```

## 测试

```bash
python -m pytest tests/ -q
```

覆盖：Skill 解析/匹配/注入、多 Agent 路由/降级/并行协作、意图识别投票、EchoGuard 安全策略与端到端代理。

## 项目结构

```text
api/              FastAPI 入口、路由、CLI
agents/           多 Agent 编排（四领域 Agent + 路由）
core/             意图识别（三路融合）、Skill 加载器
memory/           三级记忆（工作 / 情景 / 用户画像）
mcp/              RAG 知识库（ChromaDB）
evaluation/       LLM-as-Judge 评测框架
echoguard/        Agent 运行时安全 Sidecar
skills/           四领域业务 SOP（可热加载）
frontend/         Vue 3 调试面板
config/           Prometheus 等运行配置
data/             demo 知识数据 / ChromaDB 持久化
docs/             文档与历史归档
```
