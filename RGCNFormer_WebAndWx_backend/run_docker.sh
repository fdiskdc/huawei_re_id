#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/backend.env}"

# 国内镜像默认值；均可在 shell 或 backend.env 中覆盖。
export PYTHON_IMAGE="${PYTHON_IMAGE:-docker.m.daocloud.io/library/python:3.11-slim}"
export UV_VERSION="${UV_VERSION:-0.11.28}"
export REDIS_IMAGE="${REDIS_IMAGE:-docker.m.daocloud.io/library/redis:alpine}"
export DEBIAN_MIRROR="${DEBIAN_MIRROR:-https://mirrors.aliyun.com/debian}"
export DEBIAN_SECURITY_MIRROR="${DEBIAN_SECURITY_MIRROR:-https://mirrors.aliyun.com/debian-security}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

if ! command -v docker >/dev/null 2>&1; then
    echo "❌ 错误: 未找到 docker 命令"
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "❌ 错误: 未安装 Docker Compose 插件"
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $ENV_FILE"
    exit 1
fi

# --- 1. 微信敏感信息 (必填，无默认值) ---
if [ -z "${WX_APPID:-}" ]; then
    read -r -p "请输入 WX_APPID (必填): " WX_APPID
    if [ -z "$WX_APPID" ]; then
        echo "❌ 错误: WX_APPID 不能为空！"
        exit 1
    fi
    export WX_APPID
fi

if [ -z "${WX_SECRET:-}" ]; then
    read -r -s -p "请输入 WX_SECRET (必填): " WX_SECRET
    echo
    if [ -z "$WX_SECRET" ]; then
        echo "❌ 错误: WX_SECRET 不能为空！"
        exit 1
    fi
    export WX_SECRET
fi

# --- 2. 性能参数 (根据您的要求，现在也是必填，无默认值) ---
if [ -z "${GUNICORN_WORKERS:-}" ]; then
    read -r -p "请输入 GUNICORN_WORKERS 数量 (建议根据 CPU 核心数填写，如 1 或 2): " GUNICORN_WORKERS
fi

if ! [[ "$GUNICORN_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "❌ 错误: GUNICORN_WORKERS 必须是正整数"
    exit 1
fi
export GUNICORN_WORKERS

if [ -z "${CELERY_CONCURRENCY:-}" ]; then
    read -r -p "请输入 CELERY_CONCURRENCY 数量 (建议填写 1): " CELERY_CONCURRENCY
fi

if ! [[ "$CELERY_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
    echo "❌ 错误: CELERY_CONCURRENCY 必须是正整数"
    exit 1
fi
export CELERY_CONCURRENCY

# --- 3. 启动确认 ---
echo "---------------------------------------"
echo "🚀 所有必要参数已就绪，准备启动..."
echo "   - WX_APPID: $WX_APPID"
echo "   - GUNICORN_WORKERS: $GUNICORN_WORKERS"
echo "   - CELERY_CONCURRENCY: $CELERY_CONCURRENCY"
echo "   - Python 镜像: $PYTHON_IMAGE"
echo "   - uv 版本: $UV_VERSION"
echo "   - Redis 镜像: $REDIS_IMAGE"
echo "   - PyPI: $UV_DEFAULT_INDEX"
echo "---------------------------------------"

# 使用脚本所在目录作为项目目录，确保从任意路径执行都能找到构建文件。
cd "$SCRIPT_DIR"
docker compose \
    --env-file "$ENV_FILE" \
    -f docker-compose.yml \
    up -d --build
