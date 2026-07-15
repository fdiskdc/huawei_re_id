#!/bin/bash
###
 # @Author: Chao Deng && chaodeng987@outlook.com
 # @Date: 2026-02-25 03:34:03
 # @LastEditors: Chao Deng && chaodeng987@outlook.com
 # @LastEditTime: 2026-02-25 03:57:02
 # @FilePath: /backend/run_docker.sh
 # @Description: 
 # 那只是一场游戏一场梦
 #  
 # https://orcid.org/0009-0009-8520-1656
 # DOI: 10.3390/app15158626
 # DOI: 10.3390/rs17142354
 # Copyright (c) 2026 by ${Chao Deng}, All Rights Reserved. 
### 


#!/bin/bash

# --- 1. 微信敏感信息 (必填，无默认值) ---
if [ -z "$WX_APPID" ]; then
    read -p "请输入 WX_APPID (必填): " WX_APPID
    if [ -z "$WX_APPID" ]; then
        echo "❌ 错误: WX_APPID 不能为空！"
        exit 1
    fi
    export WX_APPID
fi

if [ -z "$WX_SECRET" ]; then
    read -p "请输入 WX_SECRET (必填): " WX_SECRET
    if [ -z "$WX_SECRET" ]; then
        echo "❌ 错误: WX_SECRET 不能为空！"
        exit 1
    fi
    export WX_SECRET
fi

# --- 2. 性能参数 (根据您的要求，现在也是必填，无默认值) ---
if [ -z "$GUNICORN_WORKERS" ]; then
    read -p "请输入 GUNICORN_WORKERS 数量 (建议根据 CPU 核心数填写，如 1 或 2): " GUNICORN_WORKERS
    if [ -z "$GUNICORN_WORKERS" ]; then
        echo "❌ 错误: GUNICORN_WORKERS 不能为空！"
        exit 1
    fi
    export GUNICORN_WORKERS
fi

if [ -z "$CELERY_CONCURRENCY" ]; then
    read -p "请输入 CELERY_CONCURRENCY 数量 (建议填写 1): " CELERY_CONCURRENCY
    if [ -z "$CELERY_CONCURRENCY" ]; then
        echo "❌ 错误: CELERY_CONCURRENCY 不能为空！"
        exit 1
    fi
    export CELERY_CONCURRENCY
fi

# --- 3. 启动确认 ---
echo "---------------------------------------"
echo "🚀 所有必要参数已就绪，准备启动..."
echo "   - WX_APPID: $WX_APPID"
echo "   - GUNICORN_WORKERS: $GUNICORN_WORKERS"
echo "   - CELERY_CONCURRENCY: $CELERY_CONCURRENCY"
echo "---------------------------------------"

# 使用 docker-compose 启动并强制重新构建
docker compose up -d --build