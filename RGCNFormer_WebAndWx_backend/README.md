# mRModN_WebAndWx_backend

[中文](#中文) | [English](#english)

---

<a name="中文"></a>
# mRModN RNA 分类后端服务

## 项目简介

mRModN_WebAndWx_backend 是一个基于深度学习的 RNA 序列分类后端服务，使用图卷积网络（GCN）和类查询注意力机制实现 RNA 序列的 12 类多标签分类。支持 Web 应用和微信小程序两种前端接入方式，并提供丰富的模型可解释性功能。

## 主要特性

- **RNA 序列分类**：支持 12 类 RNA 修饰分类任务
- **深度学习模型**：结合多尺度 CNN、GCN 和 Class-Query Attention
- **异步处理**：使用 Celery 实现任务队列和后台处理
- **缓存机制**：Redis 缓存提升响应速度
- **模型可解释性**：Integrated Gradients 和 GCN 聚合可视化
- **微信小程序支持**：完整的用户登录和任务提交接口
- **Docker 支持**：一键部署，开箱即用

## 项目结构

```
mRModN_WebAndWx_backend/
├── mrmodn_backend/             # 主应用包 / Main application package
│   ├── __init__.py                # 包初始化 / Package init
│   ├── app.py                     # Flask 应用工厂 / Flask app factory
│   ├── core/                      # 核心配置模块 / Core configuration
│   │   ├── config.py              # 统一配置管理 / Unified config
│   │   ├── constants.py           # RNA 分类常量 / RNA classification constants
│   │   └── paths.py               # 资源路径解析 / Resource path resolution
│   ├── models/                    # 模型定义 / Model definitions
│   │   ├── mrmodn.py          # 主模型 (RNA_ClassQuery_Model) / Main model
│   │   ├── onnx_compatible.py     # ONNX 导出版本 / ONNX export version
│   │   └── runtime.py             # 模型加载工具 / Model loading utilities
│   ├── services/                  # 业务逻辑层 / Business logic
│   │   ├── prediction.py          # 预测预处理 / Prediction preprocessing
│   │   ├── rna_structure.py       # RNA 二级结构（LinearFold）/ RNA secondary structure
│   │   └── model_inspection.py    # 模型结构检查 / Model inspection
│   ├── api/                       # Flask API 路由 / Flask API routes
│   │   ├── health.py              # 健康检查 / Health check
│   │   ├── prediction.py          # 预测接口 / Prediction endpoints
│   │   ├── wechat.py              # 微信登录 / WeChat login
│   │   └── explainability.py      # 模型可解释性 / Model explainability
│   ├── data/                      # 数据集 / Dataset
│   │   └── human.py               # Mer100Dataset (PyG) / Human RNA dataset
│   ├── training/                  # 训练相关 / Training utilities
│   │   ├── engine.py              # 训练/评估引擎 / Training/eval engine
│   │   ├── sampling.py            # 平衡采样器 / Balanced samplers
│   │   └── metrics.py             # 评估指标 / Evaluation metrics
│   └── workers/                   # Celery 任务 / Celery tasks
│       └── tasks.py               # 异步预测任务 / Async prediction tasks
├── LinearFold/                    # RNA 二级结构预测工具（外部依赖，不可修改）
├── json/                          # 配置和数据文件
│   ├── human.json                 # 模型标签映射配置
│   └── model_graph.json           # 模型计算图
├── wsgi.py                        # WSGI 入口（Gunicorn）
├── main.py                        # 开发服务器入口
├── Dockerfile                     # Docker 构建文件
├── docker-compose.yml             # Docker 编排文件
├── pyproject.toml                 # Python 依赖声明（uv）
├── uv.lock                        # 依赖锁定文件
├── .python-version                # Python 版本约束
└── .env.example                   # 环境变量模板
```

## LinearFold 保护声明

> **`LinearFold/` 目录为外部 C++ 依赖，属于第三方工具。请勿修改该目录下的任何文件。** 所有项目自有代码位于 `mrmodn_backend/` 包内。
>
> **The `LinearFold/` directory is an external C++ dependency (third-party tool). Do NOT modify any files in this directory.** All project-owned code resides in the `mrmodn_backend/` package.

## 快速开始

### 环境要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)（Python 包管理器）
- Redis 服务器
- Docker（推荐）

### 方法 1：使用 Docker（推荐）

```bash
git clone https://github.com/fdiskdc/mRModN_WebAndWx_backend.git
cd mRModN_WebAndWx_backend

# 复制并编辑环境变量 / Copy and edit environment variables
cp .env.example .env

# 构建并启动服务 / Build and start services
docker-compose up -d

# 查看日志 / View logs
docker-compose logs -f
```

### 方法 2：本地 Flask 开发服务器

```bash
uv sync --locked

# 编译 LinearFold / Compile LinearFold
cd LinearFold && make && cd ..

# 启动 Celery Worker / Start Celery Worker
uv run celery -A mrmodn_backend.workers.mrmodn_backend.workers.tasks.celery_app worker --loglevel=info

# 启动 Flask 开发服务器 / Start Flask dev server
uv run python main.py
```

### 方法 3：Gunicorn 生产部署

```bash
uv sync --locked

# 启动 Celery Worker / Start Celery Worker
uv run celery -A mrmodn_backend.workers.mrmodn_backend.workers.tasks.celery_app worker --concurrency=1 --loglevel=info

# 使用 Gunicorn 启动 / Start with Gunicorn
uv run gunicorn -w 1 -b 0.0.0.0:8000 --timeout 120 wsgi:app
```

## API 文档

### 基础接口 / Basic Endpoints

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/mrmodn/api/health` | 健康检查 / Health check |
| POST | `/mrmodn/api/v1/submit-task` | 提交预测任务 / Submit prediction task |
| GET | `/mrmodn/api/v1/results/<job_id>` | 获取预测结果 / Get prediction result |

### 微信小程序接口 / WeChat Endpoints

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/mrmodn/api/v1/wx/login` | 微信登录 / WeChat login |
| POST | `/mrmodn/api/v1/wx-submit-task` | 批量提交（最多5条）/ Batch submit (up to 5) |
| GET | `/mrmodn/api/v1/wx-task-progress/<job_id>` | 查询批量进度 / Query batch progress |

### 模型可解释性接口 / Explainability Endpoints

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/mrmodn/api/v1/model-architecture` | 获取模型架构 / Get model architecture |
| GET | `/mrmodn/api/v1/model-graph` | 获取计算图 / Get computation graph |
| POST | `/mrmodn/api/v1/integrated-gradients` | IG 归因分析 / Integrated Gradients |
| POST | `/mrmodn/api/v1/visualize-gcn-aggregation` | GCN 聚合可视化 / GCN aggregation viz |

## 测试

```bash
# 运行单元测试 / Run unit tests
uv run pytest tests/

# 检查类型标注 / Check type annotations
# （如使用 mypy）/ (if using mypy)
uv run mypy mrmodn_backend/
```

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件

---

<a name="english"></a>
# mRModN RNA Classification Backend Service

## Project Overview

mRModN_WebAndWx_backend is a deep learning-based RNA sequence classification backend service that implements 12-class multi-label classification using Graph Convolutional Networks (GCN) and Class-Query attention mechanisms. It supports both Web application and WeChat Mini Program frontends with rich model interpretability features.

## Project Structure

```
mRModN_WebAndWx_backend/
├── mrmodn_backend/             # Main application package
│   ├── __init__.py                # Package init
│   ├── app.py                     # Flask app factory
│   ├── core/                      # Core configuration
│   │   ├── config.py              # Unified config management
│   │   ├── constants.py           # RNA classification constants
│   │   └── paths.py               # Resource path resolution
│   ├── models/                    # Model definitions
│   │   ├── mrmodn.py          # Main model (RNA_ClassQuery_Model)
│   │   ├── onnx_compatible.py     # ONNX export version
│   │   └── runtime.py             # Model loading utilities
│   ├── services/                  # Business logic
│   │   ├── prediction.py          # Prediction preprocessing
│   │   ├── rna_structure.py       # RNA secondary structure (LinearFold)
│   │   └── model_inspection.py    # Model inspection
│   ├── api/                       # Flask API routes
│   │   ├── health.py              # Health check
│   │   ├── prediction.py          # Prediction endpoints
│   │   ├── wechat.py              # WeChat login
│   │   └── explainability.py      # Model explainability
│   ├── data/                      # Dataset
│   │   └── human.py               # Mer100Dataset (PyG)
│   ├── training/                  # Training utilities
│   │   ├── engine.py              # Training/eval engine
│   │   ├── sampling.py            # Balanced samplers
│   │   └── metrics.py             # Evaluation metrics
│   └── workers/                   # Celery tasks
│       └── tasks.py               # Async prediction tasks
├── LinearFold/                    # RNA secondary structure tool (external, DO NOT MODIFY)
├── json/                          # Config and data files
├── wsgi.py                        # WSGI entry (Gunicorn)
├── main.py                        # Dev server entry
├── Dockerfile                     # Docker build file
├── docker-compose.yml             # Docker orchestration
├── pyproject.toml                 # Python dependency declaration (uv)
├── uv.lock                        # Dependency lock file
├── .python-version                # Python version constraint
└── .env.example                   # Environment variable template
```

## LinearFold Protection Notice

> **The `LinearFold/` directory is an external C++ dependency (third-party tool). Do NOT modify any files in this directory.** All project-owned code resides in the `mrmodn_backend/` package.

## Quick Start

### Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Redis server
- Docker (recommended)

### Method 1: Using Docker (Recommended)

```bash
git clone https://github.com/fdiskdc/mRModN_WebAndWx_backend.git
cd mRModN_WebAndWx_backend
cp .env.example .env
docker-compose up -d
docker-compose logs -f
```

### Method 2: Local Flask Dev Server

```bash
uv sync --locked
cd LinearFold && make && cd ..
uv run celery -A mrmodn_backend.workers.mrmodn_backend.workers.tasks.celery_app worker --loglevel=info
uv run python main.py
```

### Method 3: Gunicorn Production

```bash
uv sync --locked
uv run celery -A mrmodn_backend.workers.mrmodn_backend.workers.tasks.celery_app worker --concurrency=1 --loglevel=info
uv run gunicorn -w 1 -b 0.0.0.0:8000 --timeout 120 wsgi:app
```

## API Documentation

See the Chinese section above for the full API table.

## Testing

```bash
uv run pytest tests/
```

## ReID body heatmaps

The Web ReID tab uses the CPU-only PyTorch checkpoint at `outputs/best.pt` and
reads SYSU-MM01 without modifying it. Configure the dataset path with
`REID_DATA_ROOT` (default: `/home/dc/vscode/re_id/SYSU-MM01`). The API exposes
metadata, deterministic four-sample RGB/IR batches, and indexed sample images
under `/mrmodn/api/v1/reid`.

## License

This project is licensed under the MIT License - see the LICENSE file for details
