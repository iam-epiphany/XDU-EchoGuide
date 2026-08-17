# 西电校园助手（EchoGuide）Frontend

独立 Vue 前端项目，可同时连接 EchoGuide Python 版本和 EchoGuide Java 版本，用于校园助手的对话调试与知识库管理。

## 功能

- 在页面中切换 Java / Python 后端。
- 统一适配 `/chat` 响应字段：
  - Python：`conv_id`、`agent_type`、`latency_ms`
  - Java：`conversation_id`、`agent_type`、`latency_ms`
- 支持聊天调试、健康检查、监控摘要、知识库检索、知识库文档导入、文件上传。
- 支持 Docker + Nginx 部署。

## 默认后端地址

| 后端 | 默认地址 |
|------|----------|
| Python | `http://localhost:8100` |
| Java | `http://localhost:8080` |

开发模式下，Vite 会代理：

| 前端路径 | 代理到 |
|----------|--------|
| `/api/python` | `http://localhost:8100` |
| `/api/java` | `http://localhost:8080` |

Docker Compose 模式下，Nginx 在容器网络内转发到 EchoGuide 后端服务（`echoguide:8000`，见仓库根目录 `docker-compose.yml`），无需直连宿主机。

## 本地运行

安装依赖：

```bash
npm install
```

启动：

```bash
npm run dev
```

访问：

```text
http://localhost:5175
```

如果后端端口不是默认值，可以启动时覆盖：

```bash
VITE_PYTHON_API_URL=http://localhost:8100 \
VITE_JAVA_API_URL=http://localhost:8080 \
npm run dev
```

## Docker 部署

镜像内已包含前端构建（Dockerfile 多阶段构建），无需预先执行 `npm run build`。

仅部署前端容器（挂到仓库根目录 `docker-compose.yml` 的主栈网络上，
nginx 的 `echoguide:8000` upstream 才能解析到主栈后端）：

```bash
docker compose up -d --build
```

访问：

```text
http://localhost:5175
```

停止：

```bash
docker compose down
```

完整部署（后端 + Redis + ChromaDB + Prometheus + Nginx 统一入口）见仓库根目录
`docker-compose.yml`，统一入口为 `http://localhost:8088`。

## 后端启动参考

Python 版默认：

```text
http://localhost:8100
```

Java 版默认：

```text
http://localhost:8080
```

两个后端不需要同时启动。前端页面里选择当前要调试的后端即可。
