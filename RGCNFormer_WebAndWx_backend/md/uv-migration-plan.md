# RGCNFormer_WebAndWx_backend 迁移到 uv 的修改计划

## 1. 目标

将项目的 Python 依赖管理从 `pip + requirements.txt` 迁移到 `uv + pyproject.toml + uv.lock`，并通过 Docker 完成依赖解析、锁定、构建和运行。

迁移后应满足：

- 官方支持的运行平台固定为 `linux/amd64`。
- Docker 是唯一推荐的本地运行和生产部署方式。
- 主机不需要安装 uv、Python、Conda 或项目依赖。
- `pyproject.toml` 是依赖声明的唯一来源。
- `uv.lock` 是完整、可提交、可复现的依赖锁文件。
- 删除 `requirements.txt`，不再维护 pip 依赖清单。
- 运行镜像不包含 uv、pip 缓存和编译工具。
- 保持现有 Flask、Gunicorn、Celery、Redis、PyTorch、PyG 和 LinearFold 行为不变。

## 2. 已确认决策

| 项目 | 决策 |
| --- | --- |
| 官方运行平台 | `linux/amd64` |
| 推荐运行方式 | 仅 Docker / Docker Compose |
| 主机安装 uv | 不安装 |
| Python 版本 | 保持 Docker 当前使用的 Python 3.9 |
| 深度学习栈 | 首次迁移不升级，继续使用 CPU 版 PyTorch 2.0.1 |
| 依赖声明 | `pyproject.toml` |
| 依赖锁定 | 提交 `uv.lock` |
| `requirements.txt` | 删除 |
| 项目打包 | 不构建 Python package，按应用项目管理 |

## 3. 当前状态与迁移风险

### 3.1 当前依赖不完整

当前 `requirements.txt` 只声明了部分通用依赖，以下实际运行依赖由 Dockerfile 单独安装或未声明：

- `torch`
- `torch-geometric`
- `torch-scatter`
- `torch-sparse`
- `torch-cluster`
- `torch-spline-conv`
- `openpyxl`

迁移时必须将实际运行依赖全部纳入 `pyproject.toml`，避免继续存在 Dockerfile 隐式安装依赖的情况。

### 3.2 PyTorch/PyG 依赖源特殊

项目当前使用：

- CPU 版 PyTorch 2.0.1
- PyG 为 PyTorch 2.0.1 CPU 构建的扩展 wheel
- PyG wheel 地址：`https://data.pyg.org/whl/torch-2.0.1+cpu.html`

这些扩展 wheel 没有完整的 Linux ARM64 支持，因此锁文件和镜像构建必须以 `linux/amd64` 为目标。首次迁移不得同时升级 Torch/PyG，以降低模型兼容性风险。

### 3.3 Compose 挂载会覆盖默认虚拟环境

当前 Compose 将项目目录挂载到 `/app`。uv 默认在项目目录创建 `.venv`，因此直接使用 `/app/.venv` 会被 bind mount 覆盖。

迁移后虚拟环境固定放在 `/opt/venv`，并将 `/opt/venv/bin` 加入 `PATH`。

### 3.4 当前依赖版本范围过宽

当前依赖大多只设置最低版本。直接解析可能获得与 Python 3.9、Torch 2.0.1 或现有代码不兼容的新版本。

首次迁移采用以下策略：

- Torch/PyG 核心依赖精确固定。
- `numpy` 限制为 `<2`，避免旧 Torch 与 NumPy 2 的兼容风险。
- 其他直接依赖设置合理主版本上限。
- 所有传递依赖由 `uv.lock` 精确锁定。

## 4. 文件修改清单

| 文件 | 操作 | 说明 |
| --- | --- | --- |
| `pyproject.toml` | 新增 | 声明项目元数据、Python 版本、运行依赖、可选工具依赖和 uv 配置 |
| `uv.lock` | 新增 | 在 `linux/amd64` Docker 容器内生成并提交 |
| `Dockerfile` | 重写 | 使用固定版本 uv 镜像完成同步，运行阶段不包含 uv |
| `docker-compose.yml` | 修改 | 固定 `linux/amd64`，使用镜像内 `/opt/venv`，调整挂载策略 |
| `.dockerignore` | 修改 | 忽略本地虚拟环境、缓存、文档和无关构建文件 |
| `.gitignore` | 修改 | 明确忽略 `.venv/`、uv 缓存和本地生成文件，但不忽略 `uv.lock` |
| `README.md` | 修改 | Docker 成为唯一推荐运行方式，删除 pip/Conda 使用说明 |
| `start_backend.sh` | 删除或改造 | 不再激活 Conda；建议改为 Compose 启动入口 |
| `stop_backend.sh` | 删除或改造 | 建议改为 `docker compose down` 的简单入口 |
| `run_docker.sh` | 修改 | 保留环境变量收集，统一调用固定平台的 Compose 构建与启动 |
| `requirements.txt` | 删除 | 依赖声明完全迁移到 `pyproject.toml` |

`config.py`、`config_docker.py`、`tasks.py` 和 `tasks_docker.py` 的合并不属于本次 uv 迁移范围。除非验证发现 Docker 启动问题，否则不改动业务代码。

## 5. `pyproject.toml` 设计

### 5.1 项目配置原则

- 设置 `requires-python = "==3.9.*"`，与运行镜像一致。
- 设置 `[tool.uv] package = false`，项目不作为 Python 包安装。
- 将锁文件解析环境限制为 Linux x86_64。
- 普通依赖从标准 Python 包索引解析。
- Torch CPU wheel 和 PyG 扩展使用各自官方来源。
- 不在 Dockerfile 中执行单独的 `uv pip install`；所有 Python 依赖必须由项目配置和锁文件描述。

### 5.2 首次迁移的直接依赖

运行依赖至少包括：

```text
flask
flask-cors
requests
numpy
captum
celery
redis
gunicorn
openpyxl
torch
torch-geometric
torch-scatter
torch-sparse
torch-cluster
torch-spline-conv
```

建议的核心兼容版本基线：

```text
python == 3.9.*
torch == 2.0.1+cpu
numpy >= 1.20, < 2
torch-scatter == 2.1.2+pt20cpu
torch-sparse == 0.6.18+pt20cpu
torch-cluster == 1.6.3+pt20cpu
torch-spline-conv == 1.2.2+pt20cpu
```

`torch-geometric`、Flask、Celery、Redis、Captum 等直接依赖的最终版本约束，应在容器内首次锁定和完整验证后确定。不得仅依据“最新版本”选择版本。

### 5.3 可选依赖

以下脚本相关依赖不应默认进入生产运行环境：

- `onnx`
- `onnxruntime`

如确认仍需维护 ONNX 导出脚本，可放入单独的 `onnx` 依赖组，并只在专用容器命令中安装。

## 6. 锁文件生成方案

主机不安装 uv。所有 uv 命令均通过固定版本的官方 uv Docker 镜像执行。

锁文件生成流程：

1. 在 `pyproject.toml` 中完成依赖和源配置。
2. 使用 `linux/amd64` 的 uv 容器挂载项目目录。
3. 在容器中执行 `uv lock`。
4. 执行 `uv lock --check`，确认声明与锁文件一致。
5. 使用项目 Dockerfile 执行 `uv sync --locked --no-dev`。
6. 验证通过后提交 `uv.lock`。

示意命令：

```bash
docker run --rm \
  --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -v "$PWD:/workspace" \
  -w /workspace \
  ghcr.io/astral-sh/uv:<固定版本> \
  uv lock
```

要求：

- uv 镜像必须固定具体版本或 digest，不能使用 `latest`。
- `uv.lock` 必须提交到 Git。
- Docker 构建必须使用 `uv sync --locked`，禁止构建时自动修改锁文件。
- 锁文件生成后不得手工编辑。

## 7. Dockerfile 修改方案

采用多阶段构建。

### 7.1 依赖构建阶段

- 基础镜像固定为 `python:3.9-slim` 的具体 digest。
- 从固定版本的官方 uv 镜像复制 uv 二进制，或直接使用官方 uv Python 基础镜像。
- 设置 `UV_PROJECT_ENVIRONMENT=/opt/venv`。
- 先复制 `pyproject.toml` 和 `uv.lock`。
- 执行 `uv sync --locked --no-dev --no-install-project`。
- 使用 BuildKit cache mount 缓存 uv 下载内容。

### 7.2 应用构建阶段

- 安装编译 LinearFold 所需的系统依赖。
- 复制项目源码。
- 编译 LinearFold。
- 保持当前复制 `config_docker.py` 到 `config.py`、复制 `tasks_docker.py` 到 `tasks.py` 的行为，避免 uv 迁移改变应用装配方式。
- 将配置文件和任务模块的合并作为后续独立重构处理。

### 7.3 最终运行阶段

- 使用精简的 `python:3.9-slim`。
- 从构建阶段复制 `/opt/venv`、应用源码和编译后的 LinearFold。
- 设置：

```text
PATH=/opt/venv/bin:$PATH
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
PYTHONUNBUFFERED=1
```

- 最终镜像不得包含：

```text
uv
pip 下载缓存
build-essential
编译中间文件
```

- 保持默认 Gunicorn 启动行为不变。

## 8. Docker Compose 修改方案

backend 和 worker 服务均增加：

```yaml
platform: linux/amd64
```

推荐移除以下生产式 bind mount：

```yaml
volumes:
  - .:/app
```

原因：

- 运行内容应来自已经验证的镜像。
- 避免主机文件覆盖镜像内的配置、编译结果和依赖。
- 保证本地、CI 与生产运行一致。

如确实需要开发态热更新，应单独新增 `docker-compose.dev.yml`，只挂载源码；虚拟环境仍保留在 `/opt/venv`。

Compose 中的运行命令保持为：

```text
gunicorn -w 1 -b 0.0.0.0:8000 --timeout 120 wsgi:app
celery -A tasks_docker.celery_app worker --concurrency=1 --loglevel=info
```

首次迁移不改变进程拓扑和并发参数。

## 9. 脚本和文档修改方案

### 9.1 启停脚本

`start_backend.sh` 当前硬编码 Conda 环境，不再符合 Docker-only 决策。

建议：

- 将 `start_backend.sh` 改为调用 `docker compose up -d --build`。
- 将 `stop_backend.sh` 改为调用 `docker compose down`。
- 保留 `run_docker.sh` 作为需要交互输入微信配置和并发参数的启动入口。
- 脚本中不调用 `uv`、`python`、`pip`、`conda` 或主机上的 Gunicorn/Celery。

### 9.2 README

README 修改重点：

- 前置条件只保留 Docker 和 Docker Compose。
- 明确官方平台为 `linux/amd64`。
- 删除 `pip install -r requirements.txt` 和 Conda 启动说明。
- 增加构建、启动、日志查看、停止和重建命令。
- 增加“更新依赖”的容器化 uv 操作说明。
- 说明主机不需要安装 uv。
- 删除不存在的 `gunicorn.conf.py` 使用示例。

## 10. 实施顺序

### 阶段一：建立依赖声明

1. 新增 `pyproject.toml`。
2. 补齐代码实际使用但未声明的依赖。
3. 配置 Torch CPU 与 PyG wheel 来源。
4. 在 uv Docker 容器内生成 `uv.lock`。
5. 检查锁文件只覆盖 `linux/amd64 + Python 3.9` 目标。

完成标准：

- `uv lock --check` 成功。
- 锁文件包含全部直接依赖及其传递依赖。
- 没有从源码意外编译 PyG 扩展。

### 阶段二：迁移镜像构建

1. 将 Dockerfile 改为多阶段 uv 构建。
2. 固定 uv 和 Python 镜像版本。
3. 使用 `/opt/venv`。
4. 保持 LinearFold 编译。
5. 删除所有 pip 安装命令。

完成标准：

- `docker build --platform linux/amd64` 成功。
- 构建过程严格使用 `uv.lock`。
- 最终镜像中可以直接运行 Gunicorn、Celery 和测试命令。

### 阶段三：迁移运行入口

1. 更新 Compose 并固定 `linux/amd64`。
2. 移除默认源码 bind mount。
3. 改造启停脚本。
4. 更新 README。
5. 删除 `requirements.txt`。

完成标准：

- 用户只需 Docker 即可启动完整服务。
- 文档中不再出现推荐的 pip、Conda 或主机 uv 流程。

### 阶段四：验证与清理

1. 从无缓存状态重新构建。
2. 启动 backend、worker 和 Redis。
3. 验证健康检查、同步推理、异步任务和数据接口。
4. 验证锁文件未在构建中发生变化。
5. 检查 Git diff，确认没有业务代码或模型文件意外变化。

## 11. 验证清单

### 11.1 静态验证

```bash
docker run --rm \
  --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -v "$PWD:/workspace" \
  -w /workspace \
  ghcr.io/astral-sh/uv:<固定版本> \
  uv lock --check
```

检查：

- `requirements.txt` 已删除。
- Dockerfile 中不存在 `pip install`。
- 文档和脚本中不存在推荐的 Conda/pip 本地安装流程。
- `uv.lock` 已提交且未被忽略。

### 11.2 镜像构建验证

```bash
docker compose build --no-cache
docker compose run --rm backend python -c "import torch; print(torch.__version__)"
docker compose run --rm backend python -c "import torch_geometric; print(torch_geometric.__version__)"
docker compose run --rm backend python -c "import openpyxl"
docker compose run --rm backend ./LinearFold/bin/linearfold_c <<< "ACGUACGU"
```

预期：

- Torch 为 CPU 版 2.0.1。
- PyG 及扩展可以导入。
- `openpyxl` 可以导入。
- LinearFold 可执行。

### 11.3 自动测试

```bash
docker compose run --rm backend python -m unittest test_attention_distribution.py
```

`test_cache.py` 依赖 Redis 和服务状态，应在完整 Compose 启动后运行。

### 11.4 服务验证

```bash
docker compose up -d
docker compose ps
docker compose logs --no-color backend worker
curl http://localhost:8000/api/v1/health
```

至少验证：

- backend 正常启动。
- worker 成功连接 Redis。
- `/api/v1/health` 返回成功。
- 同步推理接口正常。
- 异步任务能够提交、执行并获取结果。
- 使用 Excel 数据的接口不因缺少 `openpyxl` 失败。

### 11.5 可复现性验证

```bash
git diff --exit-code -- uv.lock
docker compose build --no-cache
git diff --exit-code -- uv.lock
```

构建前后 `uv.lock` 必须保持不变。

## 12. 回滚方案

迁移应作为一个独立提交完成，避免与业务功能修改混合。

若迁移验证失败：

1. 保留失败日志和具体依赖解析结果。
2. 回滚迁移提交，恢复原 Dockerfile 和 `requirements.txt`。
3. 不回滚业务代码、模型文件或用户已有修改。
4. 根据失败原因单独调整依赖版本或 wheel 来源后重新执行迁移。

不能通过在 Dockerfile 中临时增加未锁定的 `pip install` 绕过失败，否则会破坏迁移目标。

## 13. 本次迁移不包含的工作

- 升级 Python 3.9。
- 升级 PyTorch 或 PyG。
- 增加 GPU/CUDA 镜像。
- 支持 Linux ARM64 原生运行。
- 将项目重构为可发布 Python 包。
- 合并本地和 Docker 配置文件。
- 合并 `tasks.py` 与 `tasks_docker.py`。
- 修改模型结构、推理逻辑、API 或前端协议。
- 在主机安装或配置 uv。

## 14. 完成定义

仅当以下条件全部满足时，迁移才视为完成：

- 项目中存在有效的 `pyproject.toml` 和 `uv.lock`。
- `requirements.txt` 已删除。
- 所有 Python 依赖均由 uv 锁定和安装。
- Dockerfile 不再使用 pip 安装项目依赖。
- 主机无需安装 uv、Python 或 Conda。
- `docker compose build --no-cache` 成功。
- backend、worker、Redis 和 LinearFold 正常工作。
- 自动测试与核心 API 验证通过。
- README 只推荐 Docker 工作流。
- 官方运行平台明确固定为 `linux/amd64`。
