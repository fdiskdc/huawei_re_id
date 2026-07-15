# ReID 改动迁移到 Cluster 分支的实施计划

## 1. 目标与已确认决策

本计划用于创建一个与现有 ReID 项目相互隔离的新工作项目：

```text
源项目（保持不变）
/home/dc/vscode/re_id

新项目
/home/dc/vscode/re_id_cluster
```

在新项目中，将 ReID 项目内前后端已经完成的全部 Git 可见改动，分别重放到以下远端 Cluster 分支：

| 子项目 | ReID 改动原始基线 | Cluster 目标分支 | 已确认远端 HEAD |
|---|---|---|---|
| 后端 | `5fd4c1b9f38224b75c999812f5beb40d980cc815` | `cluster_webandwxminigame_backend` | `14026c93641a4b48cacbc908f28e133181dfc5a7` |
| 前端 | `25f9ea7f4806a70be6ed8ee4eb1e75aae52433cc` | `Cluster_WebAndWx_WebFrontend` | `1438ea2dc97180930e917714841bf3f79a0139cc` |

所有 GitHub `clone`、`fetch`、`pull` 操作使用本机 HTTP 代理：

```text
http://127.0.0.1:7897
```

代理只通过单条 Git 命令的 `-c http.proxy=...` 和 `-c https.proxy=...` 注入，不写入全局 Git 配置，也不传入 Podman/Docker 构建环境。

本计划锁定以下原则：

- `/home/dc/vscode/re_id` 始终作为只读迁移源，不在其中执行 `git add`、`git commit`、`git checkout`、`git reset`、`git rebase` 或文件覆盖。
- 不直接覆盖 Cluster 分支的整个目录；Cluster 的现有架构和部署语义优先。
- ReID 未提交改动先在临时分支上转换为可审查的提交，再使用真正的 `git rebase --onto` 重放到 Cluster HEAD。
- 不修改 `/home/dc/RGCNFormer_WebAndWx_backend` 和 `/home/dc/RGCNFormer_WebAndWx_WebFrontend` 当前已有的 `fusion` 工作树及其未提交改动。
- 不自动 push，不 force-push，不改写远端 Cluster 分支。
- `uv.lock`、`package-lock.json` 和 `dist/` 不做机械 ours/theirs 选择，而是在合并后的 Cluster 源码上重新生成。

## 2. Git 边界设计

### 2.1 迁移阶段

迁移过程中，`re_id_cluster` 顶层仓库与两个子项目临时形成三套 Git 边界：

```text
/home/dc/vscode/re_id_cluster/.git
├── RGCNFormer_WebAndWx_backend/.git
└── RGCNFormer_WebAndWx_WebFrontend/.git
```

两个子项目必须是从各自 Cluster 分支直接拉取的独立仓库，才能在其中执行分支创建、提交和 rebase。

### 2.2 验收后的默认形态

默认在两个子项目完成 rebase、测试和迁移证据归档后：

1. 为两个子项目分别创建 `git bundle` 和 `format-patch` 备份。
2. 只删除 `re_id_cluster` 中两个子项目的 `.git` 元数据。
3. 将最终前后端文件作为普通目录交给 `re_id_cluster/.git` 统一管理。

最终结构恢复为单一仓库：

```text
/home/dc/vscode/re_id_cluster/.git
├── RGCNFormer_WebAndWx_backend/
└── RGCNFormer_WebAndWx_WebFrontend/
```

这样既能证明前后端确实基于 Cluster 分支完成 rebase，又不会让最终项目重新陷入嵌套仓库状态。

如果后续明确要求前后端长期各自执行 `git pull`，则停止上述“移除子项目 `.git`”步骤，把 `re_id_cluster` 定义为多仓库工作区；此时顶层仓库不应再直接跟踪两个子目录，需要改为 submodule、subtree 或纯工作区清单。两种模式不能混用。

## 3. 执行前门禁

### 3.1 路径门禁

执行前必须确认：

```bash
realpath /home/dc/vscode/re_id
test ! -e /home/dc/vscode/re_id_cluster
```

若 `re_id_cluster` 已存在，立即停止，禁止使用 `rm -rf` 或 `rsync --delete` 覆盖未知内容。

### 3.2 代理门禁

当前制定计划时，WSL 中没有检测到 `7897` 端口监听。正式执行前必须先启动代理并通过以下检查：

```bash
ss -ltn | grep ':7897'
curl --proxy http://127.0.0.1:7897 -I https://github.com
```

如果代理实际运行在 Windows 主机但 WSL 无法通过 `127.0.0.1` 访问，应先修复 WSL mirrored networking 或代理监听地址；不在迁移过程中擅自改用其他端口。

定义仅供本次命令使用的代理包装：

```bash
gitp() {
  git \
    -c http.proxy=http://127.0.0.1:7897 \
    -c https.proxy=http://127.0.0.1:7897 \
    "$@"
}
```

禁止使用：

```bash
git config --global http.proxy ...
git config --global https.proxy ...
```

### 3.3 源项目不变性快照

在任何复制操作之前保存以下只读证据：

- 顶层仓库当前分支、HEAD、`git status --porcelain=v1`。
- 顶层已跟踪修改的 binary diff 哈希。
- 两个子目录相对原独立仓库基线的完整状态。
- 所有 Git 可见新增文件的相对路径和 SHA-256。
- 两个脚本的文件模式，尤其是 `run_docker.sh` 和 `scripts/deploy/run_docker.sh`。

源子项目的原 Git 元数据位于：

```text
/home/dc/vscode/re_id-nested-git-backup-20260715-090347/
├── RGCNFormer_WebAndWx_backend.git
└── RGCNFormer_WebAndWx_WebFrontend.git
```

迁移结束后重新生成同一份快照并逐项比较。任何差异都视为“不满足保持 re_id 不变”。

## 4. 创建独立的 re_id_cluster

### 4.1 克隆顶层 Git 历史

使用本地独立 clone，而不是简单复制 `.git`：

```bash
git clone --no-hardlinks \
  /home/dc/vscode/re_id \
  /home/dc/vscode/re_id_cluster
```

`--no-hardlinks` 确保新旧仓库不共享可变 Git 对象文件。

本地 clone 只包含已提交状态，因此还需要下一步覆盖 ReID 当前工作树中的非提交内容和运行资产。

### 4.2 覆盖当前工作树资产

从源项目向新项目执行一次无删除的 `rsync`，但排除：

- 所有 `.git/`
- 两个即将由 Cluster clone 替换的前后端目录
- `.venv/`
- `node_modules/`
- `__pycache__/`
- `.pytest_cache/`
- 其他可重建缓存

保留复制：

- `SYSU-MM01/`
- `outputs/`
- 根目录训练与推理代码
- `old/`
- `md/`
- 其他 ReID 运行资产

示意命令：

```bash
rsync -a \
  --exclude='.git/' \
  --exclude='RGCNFormer_WebAndWx_backend/' \
  --exclude='RGCNFormer_WebAndWx_WebFrontend/' \
  --exclude='.venv/' \
  --exclude='node_modules/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  /home/dc/vscode/re_id/ \
  /home/dc/vscode/re_id_cluster/
```

禁止使用 `--delete`，避免因排除规则错误删除新 clone 中的内容。

## 5. 在新项目中直接拉取 Cluster 分支

### 5.1 后端

先安全移除新项目 clone 自带的后端普通目录，然后通过代理直接 clone 后端 Cluster 分支：

```bash
gitp clone \
  --branch cluster_webandwxminigame_backend \
  --single-branch \
  https://github.com/fdiskdc/RGCNFormer_WebAndWx_backend.git \
  /home/dc/vscode/re_id_cluster/RGCNFormer_WebAndWx_backend
```

进入仓库后再次显式执行 fast-forward-only pull：

```bash
gitp -C /home/dc/vscode/re_id_cluster/RGCNFormer_WebAndWx_backend \
  pull --ff-only origin cluster_webandwxminigame_backend
```

为了取得 ReID 改动的原始基线，再获取 `main` 历史：

```bash
gitp -C /home/dc/vscode/re_id_cluster/RGCNFormer_WebAndWx_backend \
  fetch origin main
```

### 5.2 前端

```bash
gitp clone \
  --branch Cluster_WebAndWx_WebFrontend \
  --single-branch \
  https://github.com/fdiskdc/RGCNFormer_WebAndWx_WebFrontend.git \
  /home/dc/vscode/re_id_cluster/RGCNFormer_WebAndWx_WebFrontend
```

```bash
gitp -C /home/dc/vscode/re_id_cluster/RGCNFormer_WebAndWx_WebFrontend \
  pull --ff-only origin Cluster_WebAndWx_WebFrontend
```

```bash
gitp -C /home/dc/vscode/re_id_cluster/RGCNFormer_WebAndWx_WebFrontend \
  fetch origin main
```

### 5.3 拉取后验证

必须验证实际分支和远端 HEAD，不以分支名存在作为成功标准：

```text
backend HEAD == origin/cluster_webandwxminigame_backend
frontend HEAD == origin/Cluster_WebAndWx_WebFrontend
```

如果远端 HEAD 相比本计划记录的提交已经前进，使用执行时最新的远端 Cluster HEAD 作为 rebase 目标，同时在迁移报告中记录新旧提交号。

## 6. 将 ReID 工作树改动转换为提交

ReID 子目录当前已取消嵌套 `.git`，其改动不能直接在原目录提交。本阶段只读取原目录，并在 `re_id_cluster` 的两个独立子仓库中重建迁移提交。

### 6.1 后端迁移分支

在后端仓库从精确基线创建迁移分支：

```bash
git -C /home/dc/vscode/re_id_cluster/RGCNFormer_WebAndWx_backend \
  switch --create migration/reid-source-backend \
  5fd4c1b9f38224b75c999812f5beb40d980cc815
```

从以下源工作树提取 tracked 修改、删除、权限变化和 non-ignored 新文件：

```text
/home/dc/vscode/re_id/RGCNFormer_WebAndWx_backend
```

源状态至少包含：

- ReID API、模型、runtime、dataset、heatmap service。
- ReID 单元测试和集成 API 测试。
- `.env.example`、配置、路径、README。
- Dockerfile、Compose、启动脚本及可执行位。
- `pyproject.toml` 和 `uv.lock` 的依赖及镜像源更新。

建议拆分为：

1. `feat(reid): add models, runtime, dataset and heatmap services`
2. `feat(api): expose ReID visualization endpoints and configuration`
3. `test(reid): add visualization unit and integration coverage`
4. `build(deploy): add ReID dependencies and container configuration`

### 6.2 前端迁移分支

```bash
git -C /home/dc/vscode/re_id_cluster/RGCNFormer_WebAndWx_WebFrontend \
  switch --create migration/reid-source-frontend \
  25f9ea7f4806a70be6ed8ee4eb1e75aae52433cc
```

从以下源工作树重建改动：

```text
/home/dc/vscode/re_id/RGCNFormer_WebAndWx_WebFrontend
```

建议拆分为：

1. `feat(reid): add visualization page and heatmap canvas`
2. `feat(reid): integrate routes, API client and translations`
3. `build(frontend): update proxy configuration and generated assets`

迁移新增文件时必须使用原独立仓库的 ignore 规则，只纳入 Git 可见文件，不复制 `node_modules` 或其他 ignored 内容。

## 7. 基于 Cluster 执行 rebase

### 7.1 后端

在 rebase 前保留不可变安全引用：

```bash
git -C /home/dc/vscode/re_id_cluster/RGCNFormer_WebAndWx_backend \
  branch safety/reid-source-backend migration/reid-source-backend
```

然后执行：

```bash
git -C /home/dc/vscode/re_id_cluster/RGCNFormer_WebAndWx_backend \
  rebase \
  --onto origin/cluster_webandwxminigame_backend \
  5fd4c1b9f38224b75c999812f5beb40d980cc815 \
  migration/reid-source-backend
```

完成后将分支命名为：

```text
migration/reid-backend-on-cluster
```

### 7.2 前端

```bash
git -C /home/dc/vscode/re_id_cluster/RGCNFormer_WebAndWx_WebFrontend \
  branch safety/reid-source-frontend migration/reid-source-frontend
```

```bash
git -C /home/dc/vscode/re_id_cluster/RGCNFormer_WebAndWx_WebFrontend \
  rebase \
  --onto origin/Cluster_WebAndWx_WebFrontend \
  25f9ea7f4806a70be6ed8ee4eb1e75aae52433cc \
  migration/reid-source-frontend
```

完成后将分支命名为：

```text
migration/reid-frontend-on-cluster
```

## 8. 冲突解决规则

Cluster 与 ReID 在所有已修改的 tracked 文件上都有重叠，必须预期发生冲突。禁止执行：

```bash
git checkout --ours .
git checkout --theirs .
git restore --ours .
git restore --theirs .
```

### 8.1 后端规则

- Cluster 目录结构、原有 RNA 功能、微信接口和部署约束必须保留。
- ReID 新增的 API、model、runtime、dataset、heatmap service 和测试必须完整接入。
- `mrmodn_backend/app.py` 采用 Cluster 当前应用工厂，再注册 ReID blueprint。
- `core/config.py` 保留 Cluster 已有变量，再增加 ReID CPU、数据集、checkpoint、线程和缓存配置。
- `core/paths.py` 基于 Cluster 当前路径体系加入 ReID 路径，不恢复已经被 Cluster 淘汰的硬编码。
- `pyproject.toml` 保留 Cluster 依赖（例如 `python-dotenv`），同时加入 ReID 依赖（例如 `pillow`）。
- Python 默认索引使用阿里云，PyTorch 使用官方 CPU wheel 索引。
- `uv.lock` 不手工拼接；先完成 `pyproject.toml`，再在最终 Cluster 树执行 `uv lock`。
- `docker-compose.yml` 保留 Cluster 网络和服务设计，再加入 ReID dataset bind mount、checkpoint、CPU 和可配置端口。
- Dockerfile 保留 Cluster 构建需求，并合入阿里云 Debian 源与项目级 uv 配置。
- 两个启动脚本必须保持可执行位。

### 8.2 前端规则

- 保留 Cluster 当前路由、布局、首页和部署前缀。
- 加入 ReID 页面、Heatmap Canvas、API client、导航入口和中英文翻译。
- 不覆盖 Cluster 的 `MainPage.tsx` 语义。
- Vite proxy 以 Cluster 配置为基准加入 ReID 所需 rewrite。
- `package-lock.json` 不手工选择一侧；完成源码和 `package.json` 后重新生成。
- `dist/assets/index-*.js` 和 `index-*.css` 不手工解决 hash 文件冲突；删除旧构建产物并从最终源码重新构建。

## 9. 生成文件重建

### 9.1 后端锁文件

```bash
cd /home/dc/vscode/re_id_cluster/RGCNFormer_WebAndWx_backend
uv lock
uv lock --check
uv sync --locked
```

检查 `uv.lock` 中普通 PyPI 包来自：

```text
https://mirrors.aliyun.com/pypi/simple/
```

PyTorch wheel 来自：

```text
https://download.pytorch.org/whl/cpu
```

### 9.2 前端锁文件与 dist

```bash
cd /home/dc/vscode/re_id_cluster/RGCNFormer_WebAndWx_WebFrontend
npm install
npm ci
npm run lint
npm run build
```

重新生成的 `package-lock.json` 和 `dist/` 作为 Cluster + ReID 最终源码的构建结果提交。

## 10. 验证计划

### 10.1 后端静态与测试验证

```bash
cd /home/dc/vscode/re_id_cluster/RGCNFormer_WebAndWx_backend
uv run pytest tests/unit/test_reid_visualization.py
uv run pytest tests/integration/test_api_routes.py
uv run pytest
podman compose config
```

### 10.2 后端容器验证

```bash
podman compose build
podman compose up -d
podman compose ps
```

验证健康接口和 ReID 接口。宿主端口以最终 Compose 配置为准，不能假定仍为 `8000`。

Git 代理不得自动注入容器构建；先前 `127.0.0.1` 代理进入构建容器会把 localhost 错误解释为容器自身。

### 10.3 前端验证

- `npm run lint` 通过。
- `npm run build` 通过。
- ReID 页面路由可访问。
- API 请求使用 Cluster 后端前缀。
- 中文和英文翻译完整。
- 四阶段热力图可以加载并切换。
- 现有 Cluster 页面没有回归。

### 10.4 rebase 证据

分别保存：

```bash
git range-diff \
  <source-base>..safety/reid-source-* \
  <cluster-head>..migration/reid-*-on-cluster
```

验收要求：

- 所有 ReID 迁移提交均位于对应 Cluster HEAD 之后。
- 新 ReID 文件没有遗漏。
- 生成文件差异能够由最终源码和构建命令解释。
- 两个子仓库工作树干净。

## 11. 迁移历史归档与顶层统一

在移除子项目 `.git` 前，将历史保存到新项目外部：

```text
/home/dc/vscode/re_id_cluster_migration_backup/
├── backend-cluster-rebase.bundle
├── backend-format-patch/
├── frontend-cluster-rebase.bundle
└── frontend-format-patch/
```

归档验证完成后：

1. 再次确认两个 `.git` 的绝对路径都位于 `re_id_cluster` 对应子目录。
2. 只移走子项目 `.git` 到上述备份目录，不直接永久删除。
3. 在 `re_id_cluster` 顶层执行 `git add`，确认没有 `160000` gitlink。
4. 将前后端最终树提交到 `re_id_cluster` 顶层分支，例如：

```text
cluster-reid
```

顶层提交应说明：

- 后端基于哪个 Cluster commit。
- 前端基于哪个 Cluster commit。
- ReID 迁移提交范围。
- lockfile 和 dist 的重建命令。
- 对应 bundle 的保存位置。

## 12. 回滚策略

任一阶段失败时：

- rebase 冲突未解决：执行 `git rebase --abort`，返回 `safety/reid-source-*`。
- 生成文件异常：只丢弃新项目相应生成文件，再从合并后的源码重新生成。
- 子项目迁移结果不可信：删除的是新项目中的临时子仓库，重新从 Cluster clone；不得触碰原 `re_id`。
- 顶层统一后需要恢复独立子仓库：从 `.bundle` clone 回对应目录。
- 代理不可用：停止所有远端操作，不切换到全局代理，也不在未知网络状态下反复 pull。

## 13. 最终验收条件

只有同时满足以下条件，迁移才算完成：

- `/home/dc/vscode/re_id` 的前后快照完全一致。
- `/home/dc/vscode/re_id_cluster` 是独立顶层仓库，不与源仓库共享 Git 对象硬链接。
- 后端迁移历史以 `cluster_webandwxminigame_backend` 最新 HEAD 为父链。
- 前端迁移历史以 `Cluster_WebAndWx_WebFrontend` 最新 HEAD 为父链。
- 所有远端拉取均通过命令级 `127.0.0.1:7897` HTTP/HTTPS 代理完成。
- ReID Git 可见修改全部迁移；ignored 运行时目录未误提交。
- 后端锁文件、测试和 Compose 配置验证通过。
- 前端 lint、TypeScript 和生产构建通过。
- 迁移分支和 bundle 已保存，具备可审计、可回滚能力。
- 未执行远端 push；是否推送及推送到哪个分支另行确认。

