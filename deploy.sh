#!/bin/bash
# =============================================================================
# EchoGuide 西电校园智慧助手 — Windows (Git Bash) 一键部署脚本
#
# 使用前提：
#   1. 已安装 Docker Desktop（https://www.docker.com/products/docker-desktop/）
#      并已启动（任务栏鲸鱼图标处于运行状态）
#   2. 在 Git Bash 中运行（本项目开发环境默认就是 Git Bash）
#   3. 镜像拉取/构建需要联网（首次会下载 chromadb 镜像和 ONNX 模型，较慢）
#
# 用法（在项目根目录 D:\Agent-Project\XDU-EchoGuide 下执行）：
#   ./deploy.sh           一键部署：检查环境 → 配置 .env → 构建镜像 → 启动 → 健康检查
#   ./deploy.sh build     只构建镜像
#   ./deploy.sh start     启动服务
#   ./deploy.sh stop      停止服务
#   ./deploy.sh restart   重启服务
#   ./deploy.sh logs      查看日志（Ctrl+C 退出）
#   ./deploy.sh status    查看服务状态
#   ./deploy.sh down      停止并删除容器（保留数据卷）
#   ./deploy.sh clean     彻底清理（删除容器和数据卷，慎用）
#   ./deploy.sh help      帮助
#
# 部署完成后访问：
#   统一入口       http://localhost:8088 （前端界面 + API，一个地址全搞定）
#   API 文档       http://localhost:8100/docs （开发者用）
#  健康检查       http://localhost:8100/health
#   Prometheus     http://localhost:9090
#   ChromaDB       http://localhost:8001/api/v1/heartbeat
# =============================================================================

set -e

PROJECT_NAME="echoguide"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# ── 颜色（Git Bash 支持 ANSI）────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { printf "${GREEN}[INFO]${NC} %s\n" "$1"; }
warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$1"; }
error() { printf "${RED}[ERROR]${NC} %s\n" "$1"; }
step()  { printf "${BLUE}[STEP]${NC} %s\n" "$1"; }

# ── 1. 检查 Docker ────────────────────────────────────────────────────────────
check_docker() {
    step "检查 Docker 环境..."
    if ! command -v docker &> /dev/null; then
        error "未找到 docker 命令。请先安装并启动 Docker Desktop。"
        error "下载地址: https://www.docker.com/products/docker-desktop/"
        exit 1
    fi
    # docker info 需要 Docker Desktop 正在运行
    if ! docker info &> /dev/null; then
        error "Docker 服务未运行。请先启动 Docker Desktop（任务栏鲸鱼图标），然后重试。"
        exit 1
    fi
    if ! docker compose version &> /dev/null; then
        error "未找到 docker compose 子命令。请升级到新版 Docker Desktop。"
        exit 1
    fi
    info "Docker 环境正常: $(docker --version) | $(docker compose version --short)"
}

# ── 2. 准备 .env ──────────────────────────────────────────────────────────────
prepare_env() {
    step "检查 .env 配置..."
    if [ ! -f "$ENV_FILE" ]; then
        warn ".env 不存在，从 .env.example 复制..."
        if [ ! -f ".env.example" ]; then
            error ".env.example 也不存在，无法生成配置。"
            exit 1
        fi
        cp .env.example .env
        info "已生成 .env，请编辑它并填写你的 API Key："
        info "  ANTHROPIC_API_KEY=你的密钥（Claude 或 DeepSeek 兼容协议）"
        info "  ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic（如用 DeepSeek）"
        warn "填写完成后重新运行 ./deploy.sh"
        exit 1
    fi

    # 检查是否还停留在占位符
    if grep -q "your_anthropic_api_key_here" "$ENV_FILE" 2>/dev/null; then
        warn ".env 中的 ANTHROPIC_API_KEY 还是占位符，请先填入真实密钥再继续。"
        warn "提示: 支持 Claude 或 DeepSeek（DeepSeek 需同时设置 ANTHROPIC_BASE_URL）"
        exit 1
    fi
    info ".env 配置已就绪（检测到已配置 API Key）"
}

# ── 3. 准备数据目录 ──────────────────────────────────────────────────────────
prepare_dirs() {
    step "准备数据目录..."
    mkdir -p data/chroma data/eval logs
    info "目录就绪: data/chroma data/eval logs"
}

# ── 4. 构建镜像 ───────────────────────────────────────────────────────────────
build_images() {
    step "构建 Docker 镜像（首次需要下载依赖，请耐心等待）..."
    docker compose -f "$COMPOSE_FILE" build
    info "镜像构建完成"
}

# ── 4.1 构建前端（需要 Node.js 环境）────────────────────────────────────────
# 前端构建产物 frontend/dist 由 nginx 容器挂载提供页面
build_frontend() {
    step "构建前端界面..."
    if ! command -v npm &> /dev/null; then
        warn "未检测到 npm，跳过前端构建。"
        if [ ! -d "frontend/dist" ]; then
            warn "frontend/dist 不存在，前端将不可用。请安装 Node.js 后运行: cd frontend && npm install && npm run build"
        else
            info "使用已有的 frontend/dist 构建产物"
        fi
        return 0
    fi
    cd frontend
    if [ ! -d "node_modules" ]; then
        info "首次构建前端，安装依赖（npm install）..."
        npm install --no-audit --no-fund
    fi
    npm run build
    cd ..
    info "前端构建完成"
}

# ── 5. 启动服务 ───────────────────────────────────────────────────────────────
start_services() {
    step "启动服务（redis + chromadb + prometheus + echoguide + nginx）..."
    docker compose -f "$COMPOSE_FILE" up -d
    info "服务已启动，等待健康检查..."
}

# ── 6. 健康检查 ───────────────────────────────────────────────────────────────
health_check() {
    step "健康检查（首次启动需下载模型，最多等待 180 秒）..."
    local ok=0
    for i in $(seq 1 36); do
        sleep 5
        if curl -sf http://localhost:8100/health > /dev/null 2>&1; then
            ok=1
            break
        fi
        printf "  %s 等待主应用就绪... (%d/180s)\r" "${BLUE}[WAIT]${NC}" "$((i * 5))"
    done
    printf "\n"

    if [ "$ok" = "1" ]; then
        info "✓ 主应用健康: http://localhost:8100/health"
    else
        warn "主应用未在 180 秒内就绪，请查看日志: ./deploy.sh logs echoguide"
        warn "常见原因：ANTHROPIC_API_KEY 无效、ChromaDB 首次下载 embedding 模型较慢"
    fi
}

# ── 状态 / 日志 / 清理 ────────────────────────────────────────────────────────
status_services() {
    docker compose -f "$COMPOSE_FILE" ps
}

view_logs() {
    local service="${1:-}"
    if [ -z "$service" ]; then
        docker compose -f "$COMPOSE_FILE" logs -f
    else
        docker compose -f "$COMPOSE_FILE" logs -f "$service"
    fi
}

stop_services() {
    step "停止服务..."
    docker compose -f "$COMPOSE_FILE" stop
    info "服务已停止（可用 ./deploy.sh start 再次启动）"
}

restart_services() {
    step "重启服务..."
    docker compose -f "$COMPOSE_FILE" restart
    info "服务已重启"
}

down_services() {
    step "停止并删除容器（保留数据卷）..."
    docker compose -f "$COMPOSE_FILE" down
    info "已删除容器，数据卷保留"
}

clean_all() {
    warn "彻底清理：将删除所有容器和数据卷（redis/chromadb/prometheus 数据全部丢失）！"
    read -r -p "确认清理？输入 yes 继续: " answer
    if [ "$answer" = "yes" ]; then
        docker compose -f "$COMPOSE_FILE" down -v
        info "清理完成"
    else
        info "清理已取消"
    fi
}

show_help() {
    cat << EOF
EchoGuide 西电校园智慧助手 — Windows 一键部署脚本

用法: ./deploy.sh [命令]

命令:
    （无参数）  一键部署：环境检查 → 配置 .env → 构建镜像 → 启动 → 健康检查
    build       只构建镜像
    start       启动服务
    stop        停止服务
    restart     重启服务
    logs        查看所有服务日志（可加服务名: ./deploy.sh logs echoguide）
    status      查看服务状态
    down        停止并删除容器（保留数据卷）
    clean       彻底清理（删除容器和数据卷）
    help        显示此帮助

部署完成后访问:
    统一入口     http://localhost:8088 （前端界面 + API）
    API 文档     http://localhost:8100/docs （开发者用）
    Prometheus  http://localhost:9090
    ChromaDB    http://localhost:8001/api/v1/heartbeat

EOF
}

# ── 主入口 ────────────────────────────────────────────────────────────────────
main() {
    case "${1:-deploy}" in
        build)
            check_docker
            prepare_dirs
            build_frontend
            build_images
            ;;
        start)
            check_docker
            prepare_env
            prepare_dirs
            start_services
            health_check
            ;;
        deploy)
            check_docker
            prepare_env
            prepare_dirs
            build_frontend
            build_images
            start_services
            health_check
            info "部署完成！统一入口（前端界面）: http://localhost:8088  ·  API 文档: http://localhost:8100/docs"
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        logs)
            check_docker
            view_logs "$2"
            ;;
        status)
            check_docker
            status_services
            ;;
        down)
            down_services
            ;;
        clean)
            clean_all
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
