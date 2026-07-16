# Cluster 可视化系统后端 / Cluster Visualization Backend

## 项目背景 / Project Background

本项目是 RGCNFormer RNA 修饰分类系统的**后端服务**,为网页前端与微信小程序提供 HTTP API、模型推理、异步任务调度以及 RNA 二级结构预测能力。

This project is the **backend service** of the RGCNFormer RNA modification classification system, providing HTTP API, model inference, async task scheduling, and RNA secondary structure prediction for the web frontend and WeChat mini-program.

## 项目作用 / Purpose

**核心能力** / **Core capabilities**:
- 接收前端提交的 RNA 序列,触发模型推理(单条或批量)
- Accept RNA sequences from frontend, trigger model inference (single or batch)
- 提供同步(ONNX)与异步(Celery)两种推理模式
- Provide both synchronous (ONNX) and asynchronous (Celery) inference modes
- 集成 LinearFold 进行 RNA 二级结构预测
- Integrate LinearFold for RNA secondary structure prediction
- 提供可视化所需的注意力权重、定位概率、分类结果等数据
- Expose attention weights, localization probabilities, and classification results for visualization

## 技术栈 / Tech Stack

- Python 3.8+
- Flask + Gunicorn(WSGI HTTP 服务)/ Flask + Gunicorn (WSGI HTTP server)
- Celery + Redis(异步任务队列)/ Celery + Redis (async task queue)
- ONNX Runtime(模型推理)/ ONNX Runtime (model inference)
- PyTorch(模型定义与训练导出)/ PyTorch (model definition & training export)
- LinearFold(第三方 C++ 库,RNA 二级结构)/ LinearFold (3rd-party C++ lib, RNA secondary structure)
- Docker / docker-compose(容器化部署)/ Docker (containerized deployment)

## 目录结构 / Directory Layout

```
Cluster_WebAndWx_backend/
├── server.py               # Flask HTTP API 入口(主路由)
├── main_model.py           # PyTorch 模型推理
├── main_model_onnx.py      # ONNX Runtime 推理变体
├── common.py               # 公共常量与工具
├── human.py                # Human RNA 数据集处理(序列解析、k-mer、构图)
├── tasks.py                # Celery 异步任务
├── tasks_docker.py         # Docker 环境下的 Celery 任务变体
├── config.py               # 路径、端口、模型路径等配置
├── config_docker.py        # Docker 环境下的配置变体
├── wsgi.py                 # WSGI 入口(Gunicorn 加载)
├── onnx.py                 # PyTorch → ONNX 导出脚本
├── onnx2.py                # ONNX 导出变体(支持更多模型)
├── check.py                # LinearFold edge 诊断脚本
├── check_speed.py          # CPU 推理速度基准测试
├── test_cache.py           # Redis 缓存 + SHA256 jobId 测试
├── Dockerfile              # Docker 镜像构建
├── docker-compose.yml      # 多服务编排(backend + celery + redis)
├── .env.example            # 环境变量示例
├── requirements.txt        # Python 依赖
├── start_backend.sh        # 一键启动脚本
├── stop_backend.sh         # 停止脚本
├── run_docker.sh           # Docker 启动脚本
├── epoch_040.pt            # 已训练模型权重(13.8 MB)
├── data/                   # 运行时数据
├── json/                   # 缓存/中间结果(JSON)
└── LinearFold/             # 第三方 RNA 二级结构预测库(子模块)
    ├── src/                # C++ 源码
    └── gflags.py           # 参数解析
```

## 启动方式 / Getting Started

### 方式一:本地直接运行(开发模式) / Option 1: Local Dev Run

#### 1. 环境要求 / Prerequisites
- Python 3.8+
- Redis(异步任务需要)/ Redis (for async tasks)
- 已训练模型权重(默认 `epoch_040.pt`)

#### 2. 安装依赖 / Install
```bash
cd Cluster_WebAndWx_backend
pip install -r requirements.txt
```

#### 3. 配置环境变量 / Configure
```bash
# 直接编辑统一的可见配置文件 backend.env：
#   REDIS_HOST=localhost
#   MODEL_CHECKPOINT_PATH=./epoch_040.pt
#   WX_APPID=...
#   WX_SECRET=...
```

#### 4. 启动后端 / Start Backend
```bash
# 方式 a: 一键脚本 / one-click script
./start_backend.sh

# 方式 b: 手动 / manual
# 终端 1: 启动 Redis / start redis
redis-server

# 终端 2: 启动 Flask(Gunicorn)/ start flask
gunicorn -c gunicorn.conf.py wsgi:app

# 终端 3: 启动 Celery worker / start celery
celery -A tasks worker --loglevel=info
```

停止:/ Stop:
```bash
./stop_backend.sh
```

### 方式二:Docker 部署(生产模式) / Option 2: Docker Production

```bash
cd Cluster_WebAndWx_backend
./run_docker.sh
# 或手动:
docker-compose up -d
```

服务端口(默认):/ Ports (default):
- Flask: 8000
- Celery: 异步后台进程 / Celery: async background
- Redis: 6379

## 关键文件说明 / Key Files

| 文件 / File | 作用 / Purpose |
|---|---|
| `server.py` | Flask HTTP API 入口,提供 `/api/v1/submit-task`、`/api/v1/get-result`、`/api/v1/wx-login` 等路由;支持 IG、UMAP、注意力可视化接口 / Flask API entry with /api/v1/* routes, supports IG, UMAP, attention viz |
| `main_model.py` | PyTorch 推理:加载 `epoch_040.pt`,运行 forward,返回 logits / 概率 / 注意力 / PyTorch inference: load weights, forward pass, return logits / probs / attention |
| `main_model_onnx.py` | ONNX Runtime 推理(生产环境推荐,启动更快)/ ONNX inference (prod-recommended, faster startup) |
| `common.py` | 公共常量(NUCLEOTIDE_MAP、k-mer 映射)、工具函数(softmax、序列编码) / Common constants, utilities (softmax, sequence encoding) |
| `human.py` | Human RNA 数据处理:LinearFold 集成(`run_linearfold`)、二级结构构图(`build_edge_index_from_structure`)、`Mer100Dataset` / Human RNA data: LinearFold integration, structure-based graph construction |
| `tasks.py` | Celery 异步任务:长时间推理任务(`run_prediction_task`),Redis 缓存结果 / Celery async tasks, Redis-cached results |
| `tasks_docker.py` | Docker 环境下的 Celery 任务变体(REDIS_HOST 默认指向 docker 服务名 `redis`) / Docker-flavored Celery tasks |
| `config.py` | 全局配置:模型路径、Redis 地址、LinearFold 路径、日志级别 / Global config: model path, Redis host, LinearFold path, log level |
| `config_docker.py` | Docker 配置:`REDIS_HOST=redis`、容器内路径 / Docker config: `REDIS_HOST=redis`, container paths |
| `wsgi.py` | WSGI 入口:Gunicorn 加载此文件 / WSGI entry loaded by Gunicorn |
| `onnx.py` | PyTorch → ONNX → graph JSON 导出(供前端可视化)/ PyTorch → ONNX → graph JSON export |
| `onnx2.py` | 第二种 ONNX 导出变体(`torch.jit.trace`)/ Second ONNX export variant |
| `check.py` | LinearFold 边构建正确性诊断 / LinearFold edge construction diagnostic |
| `check_speed.py` | CPU 推理速度基准(用于性能调优)/ CPU inference speed benchmark |
| `test_cache.py` | Redis 缓存 + SHA256 jobId 一致性测试 / Redis cache + SHA256 jobId consistency test |
| `epoch_040.pt` | 已训练模型权重 / Trained model weights |

## API 端点(简述) / API Endpoints (Summary)

| 方法 / Method | 路径 / Path | 作用 / Purpose |
|---|---|---|
| `POST` | `/api/v1/submit-task` | 提交单条推理任务 / Submit single inference task |
| `POST` | `/api/v1/wx-submit-task` | 提交批量任务(微信端用)/ Submit batch task (WeChat) |
| `GET`  | `/api/v1/get-result?jobId=xxx` | 同步查询任务结果 / Sync query task result |
| `GET`  | `/api/v1/wx-get-result?jobId=xxx` | 微信端轮询 / WeChat polling |
| `POST` | `/api/v1/wx-login` | 微信登录 / WeChat login |
| `GET`  | `/api/v1/health` | 健康检查 / Health check |
| `POST` | `/api/v1/ig` | Integrated Gradients 归因 / Integrated Gradients attribution |
| `POST` | `/api/v1/umap` | UMAP 嵌入 / UMAP embedding |

## 与其他项目的关系 / Relation to Other Projects

- **rgcnformer_sum** - 模型训练与导出,`epoch_040.pt` 与 ONNX 模型由本项目加载 / Model training & export; weights loaded by this project
- **Cluster_WebAndWx_WxFrontend** - 微信前端,调用 `/api/v1/wx-*` 接口 / WeChat frontend calling /api/v1/wx-* endpoints
- **RGCNFormer_WebAndWx_WebFrontend** - 网页前端,调用 `/api/v1/*` 接口 / Web frontend calling /api/v1/* endpoints

## 注意事项 / Notes

- **LinearFold** 为第三方 C++ 库,需要编译(见其 `src/` 中的构建说明);如无需二级结构可跳过 / LinearFold is 3rd-party C++; compile per its README or skip if structure not needed
- 启动前确保 `epoch_040.pt` 与 `config.py` 中的 `MODEL_PATH` 一致 / Ensure `epoch_040.pt` matches `MODEL_PATH` in `config.py`
- Celery 需要 Redis 7+ / Celery requires Redis 7+
- 生产环境建议使用 ONNX 推理(`main_model_onnx.py`),启动比 PyTorch 快 ~5× / In prod, prefer ONNX runtime (5× faster startup)

## 详细文档 / Detailed Documentation

每个 .py 文件的顶部已添加**中英双语**功能说明,包含输入/输出/数据流/相关文件。/ Each .py file has a **bilingual** header with inputs/outputs/data flow/related files.

## ReID 人体热力图 / ReID Body Heatmaps
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
# 编辑 backend.env 后启动
docker compose --env-file backend.env up -d
docker compose logs -f
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
