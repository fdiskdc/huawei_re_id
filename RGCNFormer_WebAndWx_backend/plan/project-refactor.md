# RGCNFormer_WebAndWx_backend 重构方案

## 1. 方案摘要

本次重构在不改变现有模型算法、API 行为和部署能力的前提下完成：

1. 按职责重新组织后端代码，仅在项目根目录保留主要启动入口、工程配置、资源目录和核心代码包。
2. 为项目自有代码补充规范、有效的中英文注释和文档字符串。
3. 合并本地与 Docker 环境中的重复配置和任务代码。
4. 拆分职责过重的 `server.py`、`human.py` 和 `common.py`。
5. 建立自动化测试，保证目录迁移前后 API、模型推理和异步任务行为一致。

> **最高优先级约束：`LinearFold/` 目录及其全部内容不得修改、不得移动、不得重命名。**

---

## 2. 当前项目问题

### 2.1 根目录职责混杂

当前根目录同时包含 Flask 与 Gunicorn 启动入口、Celery 异步任务、模型定义、数据处理、训练工具、诊断脚本、导出脚本、部署脚本、JSON 资源、模型权重和第三方 `LinearFold` 工具。

### 2.2 单文件职责过重

- `server.py`：同时负责路由、Redis、微信登录、模型加载、任务查询和模型解释。
- `human.py`：同时负责标签定义、LinearFold 调用、图构建、数据集加载、缓存和手工测试。
- `common.py`：同时负责常量、配置加载、数据划分、采样器、训练、测试和指标计算。
- `tasks.py`：包含任务配置、模型加载、预测逻辑和结果缓存。

### 2.3 重复文件

- `tasks.py` 与 `tasks_docker.py` 仅有少量配置差异。
- `config.py` 与 `config_docker.py` 高度重复。
- Dockerfile 通过复制 Docker 专用文件覆盖通用文件，容易产生环境行为差异。
- `onnx.py` 与 `onnx2.py` 文件名无法表达两种导出方式的区别。

### 2.4 路径依赖当前工作目录

以下资源通过相对工作目录访问：

- `LinearFold/linearfold`
- `json/human.json`
- `json/model_graph.json`
- `epoch_040.pt`

移动 Python 文件前必须先建立稳定的项目根目录和资源路径解析方式，同时继续使用原位置的 `LinearFold/`。

### 2.5 测试覆盖不足

当前只有 `test_cache.py` 手工接口测试，没有完整的单元测试和重构回归测试，无法可靠证明迁移前后行为一致。

---

## 3. 重构目标

### 3.1 核心目标

- 根目录仅保留主要启动入口、工程配置、资源目录、核心代码包和受保护的 `LinearFold/`。
- 自有 Python 业务代码统一放入 `rgcnformer_backend/` 包。
- 诊断、导出和部署辅助脚本统一放入 `scripts/`。
- 测试统一放入 `tests/`。
- 环境差异由环境变量处理，不再维护 Docker 专用 Python 副本。
- 所有自有代码文件具有清晰、有效的中英文说明。

### 3.2 完成标准

- `LinearFold/` 的路径、文件内容和文件名与重构前完全一致。
- 根目录不再存放模型实现、业务服务、诊断或导出脚本。
- `server.py`、`wsgi.py` 和 `celery_worker.py` 仅作为薄启动入口。
- `config_docker.py` 和 `tasks_docker.py` 的重复逻辑被合并。
- 所有内部导入使用新的包路径。
- Docker、Gunicorn、Celery 和本地 Flask 启动命令均完成验证。
- 核心 API 和模型推理结果与重构前一致。
- 自有代码满足中英文注释规范。

---

## 4. 强制约束

### 4.1 必须执行

- 保持所有公开 API 路径、请求参数和响应结构不变。
- 保持模型结构、权重加载逻辑和预测计算结果不变。
- 保持 Celery 任务名称兼容，例如 `tasks.run_prediction_task`。
- 每次移动文件时同步更新导入、启动命令和测试。
- 使用环境变量区分本地、Docker 和其他部署环境。
- 对每个阶段执行自动化验证并记录结果。

### 4.2 严格禁止

- **禁止修改 `LinearFold/` 下的任何文件。**
- **禁止移动、重命名或删除 `LinearFold/` 目录。**
- 禁止修改 LinearFold 的调用参数、输入格式和输出解析逻辑。
- 禁止在目录重构中修改模型算法或预测阈值。
- 禁止删除现有功能。
- 禁止在没有行为基线测试的情况下拆分核心业务逻辑。
- 禁止机械地为每一行代码添加重复、无意义的双语注释。
- 禁止将密钥、日志、缓存或新的模型产物提交到 Git。

### 4.3 LinearFold 保护措施

重构前记录 `LinearFold/` 的 Git 状态。每个阶段结束后执行：

```bash
git diff -- LinearFold/
git status --short -- LinearFold/
```

两个命令的预期结果必须为空。

---

## 5. 目标目录结构

```text
RGCNFormer_WebAndWx_backend/
├── rgcnformer_backend/
│   ├── __init__.py
│   ├── app.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── health.py
│   │   ├── prediction.py
│   │   ├── wechat.py
│   │   └── explainability.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── logging.py
│   │   └── paths.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── rgcnformer.py
│   │   ├── onnx_compatible.py
│   │   └── runtime.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── prediction.py
│   │   ├── rna_structure.py
│   │   └── model_inspection.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── human.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sampling.py
│   │   ├── engine.py
│   │   └── metrics.py
│   └── workers/
│       ├── __init__.py
│       └── tasks.py
├── scripts/
│   ├── diagnostics/
│   │   ├── check_linearfold_edges.py
│   │   └── benchmark_inference.py
│   ├── export/
│   │   ├── export_model_graph.py
│   │   └── export_onnx_model_graph.py
│   └── deploy/
│       ├── start_backend.sh
│       ├── stop_backend.sh
│       └── run_docker.sh
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── json/
│   ├── human.json
│   └── model_graph.json
├── LinearFold/                 # 原位置保留，内容禁止修改
├── epoch_040.pt
├── server.py                   # 本地 Flask 薄启动入口
├── wsgi.py                     # Gunicorn 薄启动入口
├── celery_worker.py            # Celery 薄启动入口
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── README.md
└── project-refactor.md
```

### 5.1 根目录保留规则

根目录仅保留：

- 启动入口：`server.py`、`wsgi.py`、`celery_worker.py`
- 工程配置：`Dockerfile`、`docker-compose.yml`、`requirements.txt`
- 环境与项目文档：`.env.example`、`README.md`、`project-refactor.md`
- 当前资源：`json/`、`epoch_040.pt`
- 受保护目录：`LinearFold/`
- 主要代码包：`rgcnformer_backend/`
- 脚本目录：`scripts/`
- 测试目录：`tests/`

> 本轮重构不移动 `json/`、`epoch_040.pt` 和 `LinearFold/`，降低资源路径变更风险。

---

## 6. 文件迁移映射

| 当前路径 | 目标路径 | 处理方式 |
|---|---|---|
| `server.py` | `rgcnformer_backend/app.py` 与 `rgcnformer_backend/api/*.py` | 拆分业务实现；根目录保留薄入口 |
| `wsgi.py` | `wsgi.py` | 保留薄入口，更新应用导入 |
| 新增 | `celery_worker.py` | 新增 Celery 薄入口 |
| `tasks.py` | `rgcnformer_backend/workers/tasks.py` | 迁移并保留任务名称兼容 |
| `tasks_docker.py` | 合并到统一任务和配置模块 | 验证后删除重复文件 |
| `config.py` | `rgcnformer_backend/core/config.py` | 迁移并统一环境配置 |
| `config_docker.py` | 合并到统一配置模块 | 验证后删除重复文件 |
| `main_model.py` | `rgcnformer_backend/models/rgcnformer.py` | 仅移动并更新导入 |
| `main_model_onnx.py` | `rgcnformer_backend/models/onnx_compatible.py` | 仅移动并更新导入 |
| `human.py` | `data/human.py` 与 `services/rna_structure.py` | 分阶段拆分 |
| `common.py` | `core/constants.py` 与 `training/*.py` | 分阶段拆分 |
| `check.py` | `scripts/diagnostics/check_linearfold_edges.py` | 移动并明确命名 |
| `check_speed.py` | `scripts/diagnostics/benchmark_inference.py` | 移动并明确命名 |
| `onnx.py` | `scripts/export/export_model_graph.py` | 移动并明确命名 |
| `onnx2.py` | `scripts/export/export_onnx_model_graph.py` | 移动并明确命名 |
| `test_cache.py` | `tests/integration/test_cache_api.py` | 转为 pytest 测试 |
| `start_backend.sh` | `scripts/deploy/start_backend.sh` | 移动并更新工作目录 |
| `stop_backend.sh` | `scripts/deploy/stop_backend.sh` | 移动并更新工作目录 |
| `run_docker.sh` | `scripts/deploy/run_docker.sh` | 移动并更新工作目录 |
| `json/**` | `json/**` | 本轮保持原位置 |
| `epoch_040.pt` | `epoch_040.pt` | 本轮保持原位置 |
| `LinearFold/**` | `LinearFold/**` | **完全不处理** |

---

## 7. 中英文注释规范

### 7.1 适用范围

需要补充中英文注释的文件：

- 项目自有 Python 文件；
- 项目自有 Shell 脚本；
- `Dockerfile`；
- `docker-compose.yml`；
- `.env.example`；
- README 和新增设计文档。

不直接添加注释的文件：

- `LinearFold/` 下的全部文件；
- `.pt`、图片、PDF、二进制文件；
- JSON 数据文件。JSON 的用途由 README 或相邻文档说明。

### 7.2 Python 文件规范

每个自有 Python 文件必须包含中英文模块文档字符串：

```python
"""
中文：提供 RNA 预测相关的 API 路由。
English: Provides API routes for RNA prediction.
"""
```

公共类、公共函数、API 路由和 Celery 任务必须包含中英文 docstring：

```python
def run_prediction(sequence: str):
    """
    中文：执行单条 RNA 序列预测。
    English: Runs prediction for a single RNA sequence.
    """
```

行内双语注释只用于：

- 难以直接理解的算法步骤；
- 兼容性处理；
- 缓存一致性要求；
- 外部服务调用；
- 异常恢复和降级逻辑；
- 不能轻易修改的业务约束。

禁止添加仅重复代码含义的低价值注释。

### 7.3 注释验收

- 所有自有 Python 模块具有中英文模块 docstring。
- 所有公开函数、类、API 路由和 Celery 任务具有中英文 docstring。
- `LinearFold/` 无任何注释或内容变更。
- README 中记录目录职责、启动方法和维护约束。

---

## 8. 分阶段执行方案

### Phase 0：建立基线与保护措施

#### 工作内容

1. 记录当前 Git 状态。
2. 记录 `LinearFold/` 文件状态，确认初始状态无修改。
3. 记录当前根目录文件清单和内部导入关系。
4. 记录现有 API 路由、Celery 任务名和启动命令。
5. 使用固定 RNA 序列保存模型预测基线。
6. 保存 Docker Compose 解析结果。

#### 验收标准

- 基线报告完整。
- `LinearFold/` 初始状态明确。
- 能够比较重构前后的 API 和模型输出。

### Phase 1：建立测试基础设施

#### 工作内容

1. 新建 `tests/unit/`、`tests/integration/` 和 `tests/regression/`。
2. 将 `test_cache.py` 转换为 pytest 集成测试。
3. 增加配置加载、路径解析和 JSON 有效性测试。
4. 增加 API 路由清单测试。
5. 增加 Celery 任务注册名称测试。
6. 增加模型加载和固定输入推理回归测试。
7. 增加 `LinearFold/` 未修改检查。

#### 验收标准

- `python -m pytest tests/` 可以运行。
- 测试不需要启动完整训练流程。
- 重构前行为基线有自动化测试保护。

### Phase 2：建立包结构和稳定路径

#### 工作内容

1. 创建 `rgcnformer_backend/` 包和目标子目录。
2. 创建 `core/paths.py`，统一解析项目根目录、JSON、模型权重和 LinearFold 路径。
3. 保证 LinearFold 仍通过原始路径 `LinearFold/linearfold` 访问。
4. 创建新的薄入口，但暂不拆分业务逻辑。
5. 验证从任意工作目录启动时资源路径仍有效。

#### 验收标准

- 资源路径不再依赖启动时的当前工作目录。
- `LinearFold/` 未移动、未修改。
- 旧入口仍可使用。

### Phase 3：合并环境配置和重复任务文件

#### 工作内容

1. 将 `config.py` 和 `config_docker.py` 合并至 `core/config.py`。
2. Redis 主机、本地或 Docker 环境全部通过环境变量控制。
3. 将 `tasks.py` 和 `tasks_docker.py` 合并至 `workers/tasks.py`。
4. 保留现有 Celery 任务名称，避免调用方失效。
5. 修改 Dockerfile，不再通过复制 Docker 专用 Python 文件覆盖通用文件。
6. 更新 `docker-compose.yml` 和本地启动脚本。

#### 验收标准

- 本地和 Docker 使用同一份 Python 配置与任务代码。
- `config_docker.py` 和 `tasks_docker.py` 不再被引用。
- Celery 任务名称保持兼容。
- Docker Compose 配置有效。

### Phase 4：迁移模型、数据和公共代码

#### 工作内容

1. 将模型定义迁移到 `models/`，不改变任何模型计算逻辑。
2. 将模型加载和运行时逻辑提取到 `models/runtime.py`。
3. 将 `human.py` 中的 LinearFold 调用提取到 `services/rna_structure.py`。
4. 将数据集代码迁移到 `data/human.py`。
5. 将 `common.py` 中后端需要的常量迁移至 `core/constants.py`。
6. 将训练相关工具迁移至 `training/`。
7. 保持必要的兼容导出，避免一次性破坏所有内部导入。

#### 验收标准

- 固定输入的模型预测结果与基线一致。
- LinearFold 调用逻辑行为不变。
- `LinearFold/` 内容和位置不变。
- 所有内部导入使用包路径。

### Phase 5：拆分 Flask API

#### 工作内容

1. 引入应用工厂 `create_app()`。
2. 将健康检查、预测、微信接口和可解释性接口拆分到独立 Blueprint。
3. 将重复的单热编码、模型调用和结果处理逻辑移入服务层。
4. 根目录 `server.py` 和 `wsgi.py` 保持为薄启动入口。
5. 保持所有现有 URL、HTTP 方法、参数和响应结构不变。

#### 验收标准

- 原有 API 路由全部存在。
- API 回归测试通过。
- `server.py` 不再包含核心业务实现。

### Phase 6：整理脚本与根目录

#### 工作内容

1. 将诊断和性能脚本移至 `scripts/diagnostics/`。
2. 将 ONNX 和模型图导出脚本移至 `scripts/export/`。
3. 将部署 Shell 脚本移至 `scripts/deploy/`。
4. 更新脚本中的导入、路径和 README 命令。
5. 检查根目录仅保留定义的主要文件和目录。

#### 验收标准

- 所有脚本可以从项目根目录执行。
- 根目录不存在诊断、导出或业务实现文件。
- Docker、Gunicorn、Celery 和 Flask 启动入口有效。

### Phase 7：补充中英文注释与文档

#### 工作内容

1. 为所有自有 Python 文件添加中英文模块 docstring。
2. 为公共类、公共函数、API 路由和 Celery 任务添加中英文 docstring。
3. 为关键 Shell、Docker 和 Compose 配置增加中英文说明。
4. 清理无效、过时和仅重复代码含义的注释。
5. 更新 README 中英文项目结构、启动命令和维护说明。
6. 明确记录 `LinearFold/` 是不可修改、不可移动的第三方组件。

#### 验收标准

- 自有代码符合中英文注释规范。
- `LinearFold/` 没有任何修改。
- README 与最终目录结构一致。

### Phase 8：最终验证与清理

#### 工作内容

1. 运行所有 Python 语法检查。
2. 搜索并清除旧模块导入。
3. 运行完整 pytest 测试。
4. 验证 Docker Compose 配置。
5. 验证 Flask、Gunicorn 和 Celery 启动。
6. 对比固定输入模型输出。
7. 检查 `LinearFold/` Git 状态。
8. 检查根目录文件清单。
9. 检查 README 和注释覆盖。

#### 验收标准

- 所有测试通过。
- 没有旧模块路径引用。
- 没有未说明的行为变化。
- `LinearFold/` 零变更。
- 根目录满足精简目标。

---

## 9. 验证策略

### 9.1 每阶段必须执行

```bash
# 检查 LinearFold 未发生变化
git diff -- LinearFold/
git status --short -- LinearFold/

# 检查 Python 语法
python -m compileall rgcnformer_backend scripts tests

# 运行测试
python -m pytest tests/ -v

# 搜索旧模块导入
rg "from (main_model|human|common|config|tasks) import|import (main_model|human|common|config|tasks)"
```

### 9.2 最终启动验证

```bash
# Flask 本地入口
python server.py

# Gunicorn 入口
gunicorn -w 1 -b 0.0.0.0:8000 --timeout 120 wsgi:app

# Celery 入口
celery -A celery_worker.celery_app worker --concurrency=1 --loglevel=info

# Docker Compose 配置检查
docker compose config
```

### 9.3 行为回归验证

- 健康检查接口状态码与响应结构一致。
- Web 预测提交接口行为一致。
- 微信登录和批量任务接口行为一致。
- Celery 任务状态和 Redis 缓存键格式一致。
- 固定 RNA 序列的预测 logits、概率和类别结果一致。
- 模型结构和模型图接口行为一致。
- Integrated Gradients 与 GCN 聚合接口仍可调用。

---

## 10. 提交策略

禁止将全部重构压缩为单次提交。建议按阶段提交：

1. `test: add backend refactor regression baseline`
2. `refactor: add backend package and stable resource paths`
3. `refactor: unify runtime configuration and celery tasks`
4. `refactor: organize model data and shared modules`
5. `refactor: split flask api routes and services`
6. `refactor: organize scripts and clean project root`
7. `docs: add bilingual code documentation`
8. `docs: update backend structure and deployment guide`

每次提交前必须确认：

```bash
git diff -- LinearFold/
git status --short -- LinearFold/
python -m pytest tests/ -v
```

---

## 11. 风险与控制措施

| 风险 | 影响 | 控制措施 |
|---|---|---|
| 移动模块导致导入失效 | 服务无法启动 | 每次迁移同步更新导入并运行测试 |
| 当前工作目录变化导致资源找不到 | 模型或 LinearFold 加载失败 | 优先建立统一路径模块 |
| Celery 任务名变化 | 已提交任务无法查询 | 显式保留原任务名称 |
| Docker 与本地配置合并错误 | 部署失败 | 使用环境变量并分别验证 |
| 拆分 `server.py` 改变 API | 前端调用失败 | API 路由和响应回归测试 |
| 拆分模型代码改变预测结果 | 核心功能回归 | 固定输入模型输出对比 |
| 误改或移动 `LinearFold/` | RNA 结构预测失效 | Git 差异检查和零变更验收 |
| 注释工作制造大量噪声 | 降低可维护性 | 仅注释职责、约束和复杂逻辑 |

---

## 12. 最终验收清单

### 目录与文件

- [ ] `LinearFold/` 路径保持不变。
- [ ] `LinearFold/` 文件内容零修改。
- [ ] 根目录仅保留主要启动入口和工程级文件。
- [ ] 自有业务代码全部位于 `rgcnformer_backend/`。
- [ ] 诊断、导出和部署辅助脚本全部位于 `scripts/`。
- [ ] 测试全部位于 `tests/`。

### 行为与兼容性

- [ ] Flask 本地入口可启动。
- [ ] Gunicorn 入口可启动。
- [ ] Celery Worker 可启动。
- [ ] Docker Compose 配置有效。
- [ ] 所有现有 API 路径保持不变。
- [ ] Celery 任务名称保持兼容。
- [ ] 固定输入的模型预测结果一致。
- [ ] LinearFold 调用行为一致。

### 代码质量

- [ ] 所有 Python 文件通过语法检查。
- [ ] 所有测试通过。
- [ ] 不再存在 Docker 专用的重复 Python 文件。
- [ ] 不再存在旧模块路径引用。
- [ ] 自有 Python 文件具备中英文模块 docstring。
- [ ] 公共函数、类、API 路由和 Celery 任务具备中英文 docstring。
- [ ] README 中英文内容与最终结构一致。

---

## 13. 推荐执行顺序

```text
基线与测试
    ↓
稳定路径与包结构
    ↓
合并配置和 Celery 重复代码
    ↓
迁移模型、数据和公共代码
    ↓
拆分 Flask API
    ↓
整理脚本和根目录
    ↓
统一中英文注释
    ↓
完整回归验证
```

该顺序确保目录移动发生前已经具备路径解析和回归测试能力，并在整个重构过程中持续保护 `LinearFold/` 不被修改或移动。
