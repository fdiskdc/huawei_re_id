# ReID 人体热力图 Web 集成修改计划

## 1. 目标与已确认决策

在 `RGCNFormer_WebAndWx_WebFrontend` 中新增独立的 `reid` 可视化选项卡，将根目录 `model.py` 对应模型的四阶段人体空间热力图接入现有 Web 系统；在 `RGCNFormer_WebAndWx_backend` 中增加独立的 ReID 模型、数据读取、热力图生成和 API 服务。

本计划锁定以下决策：

- 可视化运行模式是基于 `outputs/best.pt` 的推理浏览，不启动或控制 `train.py` 训练任务，也不实时订阅训练过程。
- 后端固定采用 CPU 版 PyTorch，`REID_DEVICE=cpu`；不使用 CUDA、AMP、GPU fallback 或 ONNX Runtime。
- checkpoint 使用根项目 `outputs/best.pt`，迁移部署时复制为后端 `outputs/best.pt`，不移动或覆盖根目录原文件。
- 数据集目录由后端配置，默认值为 `/home/dc/vscode/re_id/SYSU-MM01/`。
- 一个可视化 batch 固定包含 4 个 sample，默认按“2 个身份 × 每个身份 1 张 RGB + 1 张 IR”组成。
- 页面同时展示 4 张人体热力图；批次或层发生变化时，4 个面板必须原子更新，不允许出现混合批次。
- 可视化阶段固定为 `position_embedding`、`transformer[0]`、`transformer[1]`、`classifier` 四层。
- 热力图组件的最终画布固定为 `256×256`。
- 热力图叠加到原始人体图像上；默认保持人体纵横比，不把 `144×288` 的人物图像直接拉伸成正方形。
- 第四层采用分类器预测结果引导的 Grad-CAM，展示影响身份预测的人体区域，不把 344 个分类 logits 伪装成空间热力图。
- 不修改 SYSU-MM01 数据集文件，不覆盖现有训练 checkpoint，不改变原训练模型的参数名和 checkpoint 兼容性。

## 2. 当前代码与模型基线

### 2.1 模型配置

`outputs/best.pt` 的 `model_config` 已确认如下：

| 配置 | 值 |
|---|---:|
| `num_classes` | 344 |
| `image_height` | 288 |
| `image_width` | 144 |
| `transformer_dim` | 256 |
| `transformer_depth` | 2 |
| `num_heads` | 8 |
| `embedding_dim` | 256 |

CNN 包含 5 次 stride-2 下采样，因此空间特征为：

```text
[B, 3, 288, 144]
  -> CNN
[B, 256, 9, 5]
  -> flatten patches
[B, 45, 256]
  -> add CLS
[B, 46, 256]
```

45 个 patch token 可以恢复为 `9×5` 的人体空间网格。四层人体热力图都应最终建立在这一空间对应关系上。

### 2.2 checkpoint 格式差异

ReID checkpoint 使用：

```python
checkpoint["model_config"]
checkpoint["model"]
checkpoint["label_to_pid"]
checkpoint["data_splits"]
```

当前后端 RNA 模型 loader 使用 `model_state_dict` 和另一套 `config` 结构，因此不能复用现有 `mrmodn_backend/models/runtime.py` 直接加载 ReID checkpoint。必须为 ReID 建立独立 loader。

### 2.3 前后端 API 前缀差异

当前前端公开请求前缀是 `/rgcnformer/api/v1`，后端现有路由主要使用 `/mrmodn/api/v1`，Vite 开发代理目前没有执行路径重写。

本次计划采用：

- 浏览器公开地址继续使用 `/rgcnformer/api/v1/reid/...`。
- 后端真实路由使用 `/mrmodn/api/v1/reid/...`。
- Vite 开发代理增加 `/rgcnformer` 到 `/mrmodn` 的 rewrite。
- 生产反向代理使用等价 rewrite；不在 ReID 代码中维护两套重复路由。

## 3. 四层人体热力图定义

### 3.1 Position Embedding 阶段

不能直接可视化原始 `self.position_embedding` 参数。它的 batch 维为 1，与输入 sample 无关，直接复制后会导致 4 个 sample 显示相同热力图。

应可视化加入位置编码后的实际样本 patch 特征：

```python
position_tokens = patch_tokens + position_embedding[:, 1:, :]
position_map = torch.linalg.vector_norm(position_tokens, ord=2, dim=-1)
position_map = position_map.reshape(batch_size, 9, 5)
```

该图在界面中标注为“Position-encoded feature energy”，避免将其误解为注意力权重。

### 3.2 Transformer 1 阶段

在第一个 `TransformerBlock` 的自注意力中保留注意力概率：

```text
attention: [B, 8, 46, 46]
```

提取 CLS query 对 45 个 patch key 的注意力，并对 8 个 head 求均值：

```python
transformer_0_map = attention_0[:, :, 0, 1:].mean(dim=1)
transformer_0_map = transformer_0_map.reshape(batch_size, 9, 5)
```

该热力图表达第一个 Transformer block 中 CLS token 对人体各区域的关注强度。

### 3.3 Transformer 2 阶段

使用第二个 `TransformerBlock` 的 CLS-to-patch 注意力，处理方式与 Transformer 1 相同：

```python
transformer_1_map = attention_1[:, :, 0, 1:].mean(dim=1)
transformer_1_map = transformer_1_map.reshape(batch_size, 9, 5)
```

### 3.4 Classifier 阶段

`self.classifier` 的直接输出是 `[B, 344]`，不具备图像空间结构。第四层采用 classifier-guided Grad-CAM：使用每个 sample 的预测身份 logit 对 CNN 输出特征图求梯度。

```python
feature_map = model.cnn(images)             # [B, 256, 9, 5]
feature_map.retain_grad()
embeddings, logits = visualization_forward(feature_map)
targets = logits.argmax(dim=1)              # [B]
selected = logits.gather(1, targets[:, None]).sum()
gradients = torch.autograd.grad(selected, feature_map)[0]
channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
classifier_map = torch.relu(
    (channel_weights * feature_map).sum(dim=1)
)                                             # [B, 9, 5]
```

模型始终处于 `eval()` 模式，BatchNorm 使用已保存的统计量，Dropout 关闭。仅 Grad-CAM 计算段启用 autograd；不创建 optimizer，不调用训练反向更新，也不修改模型参数。

对 4 个 sample 选中的 logits 求和后可在一次 `torch.autograd.grad` 中计算整个 batch。模型在 eval 模式下不存在 sample 间运算耦合，因此每张图的梯度仍对应自己的预测结果。

## 4. 热力图归一化、缩放与人体叠加

### 4.1 归一化范围

为了让同一批次的 4 个 sample 可以比较，同一个阶段使用 batch 级共同范围：

```text
同一 layer 的 4 张 9×5 map
  -> 共同计算低/高值范围
  -> 映射到 [0, 1]
```

- Position 和 Grad-CAM 使用 batch 级 1%/99% percentile，降低极端值对颜色范围的影响。
- Transformer attention 使用同一 layer 的 batch 级 min-max。
- 分母小于 epsilon 时返回全零图，禁止生成 NaN 或 Inf。
- 不跨 layer 共用颜色范围，因为四层数值语义和分布不同。
- API 同时返回归一化前的 `rawMin`、`rawMax`，便于调试。

### 4.2 256×256 画布与纵横比

模型输入比例是宽高 `144:288 = 1:2`。为避免人体被横向拉宽，页面使用严格的 `256×256` Canvas，但有效人体区域为约 `128×256`，水平居中：

```text
256×256 canvas
┌──────────────────────────────┐
│ padding │ 128×256 body │ padding │
└──────────────────────────────┘
```

具体流程：

1. 将原始图像按 `TestTransform` 的几何规则调整为 `144×288`。
2. 将 `9×5` 热力图先映射到完整模型输入空间。
3. 将人体图和热力图同步缩放至 `128×256`。
4. 在 `256×256` Canvas 中水平居中绘制。
5. Canvas 左右区域保持透明或使用统一深色背景。

这样组件外部尺寸满足 `256×256`，人体及热力图仍保持空间对齐。

### 4.3 显示形式

每个 sample 支持以下显示模式：

- `Overlay`：原人体图 + 彩色热力图，默认模式。
- `Heatmap`：只显示热力图。
- `Original`：只显示原人体图。

默认叠加透明度为 `0.55`，页面提供透明度滑块。颜色采用统一的蓝—青—黄—红连续色带，并显示低响应/高响应图例。

## 5. 数据集索引与四样本批次

### 5.1 默认数据目录

新增后端配置：

```env
REID_DATA_ROOT=/home/dc/vscode/re_id/SYSU-MM01
```

启动时只验证路径和必要的 `cam*`/`exp` 结构，不修改数据集。路径错误时 ReID API 返回明确的 503，不影响现有 RNA 接口注册。

### 5.2 确定性数据预处理

复用 `data_loader.py` 中 `TestTransform` 的语义：

- PIL 图像转换为 RGB；IR 图像也扩展为 3 通道。
- 调整为 `144×288`。
- 使用 ImageNet mean/std 标准化。
- 不使用随机 padding、裁剪或水平翻转。

相同 sample 在重复请求中必须得到相同输入和热力图。

### 5.3 批次构成

默认 split 为 checkpoint `data_splits["test"]`，而不是重新随机划分身份。每个 batch：

```text
PID A: 1 RGB + 1 IR
PID B: 1 RGB + 1 IR
总计: 4 samples
```

身份、camera 和图片选择按固定排序及 batch index 确定。API 可以允许 `split=test|val|train`，但 batch size 不接受前端覆盖，后端始终强制为 4。

返回 sample 元数据：

- 不透明 `sampleId`
- PID
- camera ID
- `visible`/`infrared` modality
- 数据集内相对路径
- 当前预测 class index、映射后的 PID 和分数

原图接口根据服务端索引解析 `sampleId`。禁止前端提交任意文件系统路径，防止目录穿越。

## 6. 后端修改计划

### 6.1 新增文件

1. `mrmodn_backend/models/reid.py`
   - 迁移根目录 `model.py` 的 ReID 网络定义。
   - 保持所有参数名和模块层级不变，确保 `outputs/best.pt` 可以 `strict=True` 加载。
   - 为可视化增加显式前向入口，返回 position tokens、两层 attention、CNN feature map 和 logits。
   - 普通推理 forward 行为保持不变。

2. `mrmodn_backend/models/reid_runtime.py`
   - 从 `checkpoint["model_config"]` 重建模型。
   - 从 `checkpoint["model"]` 严格加载权重。
   - 固定创建 `torch.device("cpu")`。
   - 校验 checkpoint 中必须存在 `label_to_pid` 和 `data_splits`。
   - 使用进程内锁实现惰性单例加载，避免并发首次请求重复加载模型。

3. `mrmodn_backend/services/reid_dataset.py`
   - 建立 SYSU-MM01 只读索引。
   - 实现确定性的 RGB/IR 配对和 batch index 翻页。
   - 实现安全的 sample ID 到文件路径映射。

4. `mrmodn_backend/services/reid_heatmap.py`
   - 执行 batch=4 的 CPU forward。
   - 提取四层 `9×5` 人体空间图。
   - 执行 classifier-guided Grad-CAM。
   - 按 layer 对整个 batch 归一化。
   - 将 tensor detach 后转换为小型 JSON 数组。

5. `mrmodn_backend/api/reid.py`
   - 注册 metadata、batch 和 sample image API。
   - 统一输入校验和错误 JSON。
   - 不依赖 Redis 或 Celery；批次推理是同步、只读操作。

6. `outputs/best.pt`
   - 从根目录 `outputs/best.pt` 复制，不移动源文件。
   - 复制前后记录 SHA-256，确认文件完全一致。
   - 若仓库对二进制大小有限制，使用 Git LFS 或部署时复制，不以普通 Git blob 重复提交。

### 6.2 修改文件

1. `mrmodn_backend/core/paths.py`
   - 增加后端 `outputs/best.pt` 的默认绝对路径解析。

2. `mrmodn_backend/core/config.py`
   - 增加 `REID_CHECKPOINT_PATH`、`REID_DATA_ROOT`、`REID_DEVICE`、`REID_BATCH_SIZE`、`REID_HEATMAP_SIZE` 和 CPU 线程配置。
   - `REID_DEVICE` 默认且只支持 `cpu`；收到其他值时启动校验失败并给出明确提示。

3. `mrmodn_backend/app.py`
   - 注册 `reid_bp`。
   - 不在 `create_app()` 中立即执行 ReID 模型 forward。
   - ReID 初始化失败不能破坏现有 RNA 模型服务；错误由 `/reid/meta` 和 `/reid/batches/...` 返回。

4. `.env.example`
   - 记录所有 ReID 配置及 WSL2 默认路径。

5. `pyproject.toml` 与 `uv.lock`
   - 使用 PyTorch CPU wheel 源锁定 CPU 构建。
   - 明确加入 Pillow（后端读取和返回人体图需要）。
   - 更新 lockfile 后验证安装结果不包含 CUDA runtime 依赖。

6. `docker-compose.yml`
   - 如需 Docker 运行，增加 SYSU-MM01 只读 volume，例如映射为 `/data/SYSU-MM01:ro`。
   - 容器内通过 `REID_DATA_ROOT=/data/SYSU-MM01` 覆盖 WSL 默认路径。
   - 不增加 GPU、NVIDIA runtime 或 CUDA 配置。

7. 后端测试目录
   - 增加 ReID 配置、checkpoint loader、dataset index、heatmap shape、API 和路径安全测试。

### 6.3 CPU 推理约束

- 全程 float32，不启用 autocast 或 AMP。
- 默认 Gunicorn 单 worker，与现有部署保持一致。
- 增加 `REID_TORCH_NUM_THREADS`，默认建议 4；在启动阶段设置 CPU intra-op threads。
- 使用推理锁限制同时执行的 Grad-CAM 请求，避免多个反向图同时抢占 CPU 和内存。
- 对最近批次增加小型 LRU cache，建议最多 8 个 batch；cache key 必须包含 checkpoint 修改时间、split 和 batch index。
- 自动播放必须等待上一批请求完成后再请求下一批，不使用固定间隔并发堆积请求。

## 7. API 计划

### 7.1 元数据

```http
GET /mrmodn/api/v1/reid/meta
```

返回：

- 模型是否已就绪
- checkpoint 文件名和模型配置
- CPU 设备及线程数
- 数据集目录是否可用
- 支持的 split
- batch size：4
- 原始网格：`9×5`
- 显示尺寸：`256×256`
- 四个 stage 的固定顺序

### 7.2 获取批次

```http
GET /mrmodn/api/v1/reid/batches/<batch_index>?split=test
```

一次响应返回 4 个 sample 的全部四层热力图，使前端切换 layer 时不需要重新执行 CPU 推理：

```json
{
  "batchId": "test-000012",
  "batchIndex": 12,
  "batchSize": 4,
  "gridShape": [9, 5],
  "displaySize": [256, 256],
  "stages": [
    "position_embedding",
    "transformer_0",
    "transformer_1",
    "classifier"
  ],
  "samples": [
    {
      "sampleId": "opaque-id",
      "pid": 12,
      "camera": 3,
      "modality": "infrared",
      "imageUrl": "/rgcnformer/api/v1/reid/samples/opaque-id/image",
      "prediction": {
        "classIndex": 17,
        "pid": 25,
        "score": 0.82
      },
      "heatmaps": {
        "position_embedding": {
          "values": [[0.0]],
          "rawMin": 0.0,
          "rawMax": 1.0
        },
        "transformer_0": {"values": [[0.0]]},
        "transformer_1": {"values": [[0.0]]},
        "classifier": {"values": [[0.0]]}
      }
    }
  ]
}
```

实际 `values` 必须为完整的 `9×5` 二维数组。

### 7.3 获取人体原图

```http
GET /mrmodn/api/v1/reid/samples/<sample_id>/image
```

- 只允许读取已建立索引的 SYSU-MM01 图片。
- 返回适合浏览器缓存的 ETag/Cache-Control。
- 不接受路径 query 参数。

### 7.4 错误状态

- `400`：split 或 batch index 无效。
- `404`：sample ID 不存在。
- `503`：checkpoint 或数据集不可用、ReID 服务尚未就绪。
- `500`：模型计算异常；日志保留 traceback，响应不暴露服务器绝对路径。

## 8. 前端修改计划

### 8.1 新增页面和组件

1. `src/pages/ReidViz.tsx`
   - 页面级批次、layer、播放状态和显示模式管理。
   - 加载 meta 和 batch 数据。
   - 使用同一个 batch response 构造 4 个 sample card。

2. `src/components/reid/ReidHeatmapCanvas.tsx`
   - Canvas 的逻辑宽高固定为 256。
   - 绘制保持比例的人体原图、色彩图和 overlay。
   - 将 `9×5` 值平滑插值到与人体相同的有效区域。
   - 提供 Original/Heatmap/Overlay 和透明度控制。

3. `src/pages/ReidViz.css`
   - 桌面端优先四列布局。
   - 中等宽度使用 `2×2`。
   - 手机端使用单列，Canvas 仍为 `256×256`。

### 8.2 修改现有前端文件

1. `src/App.tsx`
   - 在 `VizLayout` 下增加 `/reid` route。

2. `src/components/VizLayout.tsx`
   - 增加菜单项，显示名固定为 `reid`。

3. `src/lib/api.ts`
   - 增加 ReID meta/batch 类型定义和请求函数。
   - 对 response 的 batch size、stage keys 和二维数组 shape 做运行时检查。

4. `src/lib/i18n/zh.ts` 与 `src/lib/i18n/en.ts`
   - 增加 ReID 页面、四层名称、播放控制、模态、预测信息和错误提示翻译。

5. `vite.config.ts`
   - 为现有 `/rgcnformer/api` 代理增加到 `/mrmodn/api` 的路径 rewrite。

### 8.3 四图同步更新规则

- 一个 batch response 保存为单一不可分割状态对象。
- 禁止为 4 个 sample 分别发请求或分别更新 React state。
- 切换 layer 只改变当前 layer key，四个 Canvas 在同一次 render 中读取对应 heatmap。
- 翻页时保留当前完整 batch，直到下一批 4 个 sample 全部返回并通过 shape 校验后再替换。
- 使用 `AbortController` 取消过期请求，并用 request sequence 防止旧响应覆盖新批次。
- 自动播放只在当前请求成功后调度下一批；暂停或离开页面时清理 timer 和请求。

## 9. 实施顺序

1. 复制并校验 `outputs/best.pt`，完成 CPU ReID loader 的 strict-load 测试。
2. 完成只读 dataset index 和确定性的 4-sample RGB/IR batch。
3. 增加显式 attention 输出，验证可视化 forward 的 logits 与原 forward 一致。
4. 实现四层 `9×5` map 和 classifier-guided Grad-CAM。
5. 实现 ReID API、错误处理、推理锁和小型缓存。
6. 增加前端 route、菜单、API client 和 ReID 页面。
7. 实现 `256×256` 人体热力图 Canvas、响应式四面板和播放控制。
8. 修正开发代理 rewrite，更新环境变量、README 和 Docker CPU 部署说明。
9. 执行后端测试、前端 build、CPU 性能测试和人工视觉验收。

## 10. 验证与验收标准

### 10.1 后端自动验证

- CPU PyTorch 可导入，且 `torch.cuda.is_available()` 为 `False`。
- ReID runtime 的 device 严格等于 `cpu`。
- `outputs/best.pt` 使用 `strict=True` 无 missing/unexpected keys。
- 原始 forward 与 visualization forward 的 embeddings/logits 在允许误差内一致。
- batch 始终包含 4 个 sample，且默认构成为两个 PID 的 RGB/IR 配对。
- 四个 stage 均返回 `[4, 9, 5]`。
- 所有归一化值均为有限数并处于 `[0, 1]`。
- 两层 attention map 的来源分别对应 `transformer[0]` 和 `transformer[1]`。
- classifier Grad-CAM 非负；对正常图片至少存在一个非零响应值。
- 无效 checkpoint、无效数据目录、越界 batch 和伪造 sample ID 返回规定状态码。
- 路径穿越字符串不能访问数据集目录外文件。

### 10.2 前端自动验证

- TypeScript 编译和 Vite production build 通过。
- `/reid` route 和菜单项可访问。
- 每个 Canvas 的逻辑尺寸为 `256×256`。
- 页面同时存在且只存在 4 个当前 sample 面板。
- layer 切换不产生额外 batch API 请求。
- 过期请求不会覆盖当前 batch。
- desktop、`2×2` 和 mobile 布局不出现 Canvas 裁切。

### 10.3 人工视觉验收

- 人体未因 `256×256` 画布而横向拉伸。
- 热力图与人体头部、躯干、四肢的空间位置保持对齐。
- Position、Transformer 1、Transformer 2 和 Classifier 四层可以明确切换。
- Classifier 图显示影响身份预测的身体区域，而不是类别 logits 方阵。
- 上一批、下一批和自动播放时，4 个面板同时完成替换。
- RGB 和 IR 图片的热力图都能正常显示。
- 色条、透明度和 Original/Heatmap/Overlay 模式工作正常。

### 10.4 CPU 性能记录

在目标 WSL2 主机上记录：

- 首次模型加载耗时。
- batch=4 的完整四层计算耗时。
- Grad-CAM 耗时和峰值内存。
- 缓存命中响应耗时。

第一版不预设未经测量的硬性秒数目标。如果 CPU 单批延迟影响自动播放，优先采用缓存、限制线程和较长播放间隔，不降低 batch=4、四层语义或人体空间分辨率要求。

## 11. 风险与处理

1. Position map 可能比 attention map 更均匀。
   - 它表示加入位置编码后的特征能量，不是注意力；页面需使用准确名称和说明。

2. 暴露 attention 可能意外改变模型 forward。
   - 可视化入口必须复用相同 QKV/Softmax 结果；用 logits/embedding 一致性测试防止计算漂移。

3. CPU Grad-CAM 比纯推理慢。
   - 使用 batch 一次求梯度、单并发推理、LRU cache，并避免前端重叠请求。

4. `9×5` 网格较粗。
   - 平滑插值只能改善显示，不能创造新的模型空间信息；页面需要保留“原始网格 9×5”说明。

5. 前后端公共路径不一致。
   - 通过统一代理 rewrite 解决，避免新增双路由技术债。

6. backend checkpoint 是较大的二进制文件。
   - 复制时校验 SHA-256；版本管理使用 Git LFS 或部署制品策略。

7. 当前工作区已有用户修改和未跟踪目录。
   - 实施时保留 `outputs/last.pt`、`outputs/train_args.json` 及其他现有改动，不执行 reset/checkout 或覆盖无关文件。

## 12. 不在本次范围内

- 不修改 `train.py` 的训练循环或训练超参数。
- 不增加实时训练日志、Redis/SSE/WebSocket 训练流。
- 不重新训练或微调 `outputs/best.pt`。
- 不修改 SYSU-MM01 数据集内容。
- 不增加 CUDA、GPU Docker、AMP 或 ONNX 推理路径。
- 不将 ReID 页面接入微信小程序；本次范围仅为 WebFrontend 独立选项卡。

## 13. 完成定义

当 CPU 后端能够从指定 SYSU-MM01 目录稳定生成一个固定 4-sample batch 的四层人体空间图，Web 前端 `/reid` 页面能在 4 个 `256×256` Canvas 中同步显示、切换和播放这些人体叠加热力图，并通过上述后端、前端和人工验收项时，本次修改视为完成。
