# PyTorch CNN-Transformer 跨模态行人重识别改造计划

## 1. 目标 / Goal

在项目根目录新增一套独立的 PyTorch 实现，保留旧项目针对 SYSU-MM01 的可见光/红外配对训练协议，并提供可导出 ONNX 的 CNN + Transformer 模型。旧 MindSpore 项目与数据集不做修改。

Create an independent PyTorch implementation in the project root. Preserve the legacy visible/infrared paired-training protocol for SYSU-MM01 and provide an ONNX-exportable CNN + Transformer model. The legacy MindSpore project and dataset will remain unchanged.

## 2. 旧代码行为基线 / Legacy behavior baseline

- 数据：训练身份来自 `exp/train_id.txt` 和 `exp/val_id.txt`；可见光使用 `cam1/cam2/cam4/cam5`，红外使用 `cam3/cam6`。
- 采样：每批随机选择 P 个身份，每个身份分别从 RGB 和 IR 抽取 K 张图，形成严格对齐的跨模态 batch。
- 图像：固定为 `288 x 144`（高 x 宽），训练时使用 padding、随机裁剪、随机水平翻转和 ImageNet 标准化。
- 优化：身份分类交叉熵与跨 batch 的 triplet 度量损失联合训练；使用分组学习率、warm-up 和阶段衰减。
- 测试协议：IR query 来自 `cam3/cam6`；RGB gallery 在每个 trial 中按身份/相机随机选一张。

## 3. 新增文件 / New files

1. `data_loader.py`
   - 直接扫描原始 SYSU-MM01 目录，不再依赖旧项目中当前数据集目录里不存在的预处理 `.npy` 文件。
   - 实现训练清单、连续标签映射、RGB/IR 配对 Dataset、P×K 身份采样器、query/gallery 构建及测试 Dataset。
   - 所有公开类和关键逻辑添加中英双语注释。

2. `model.py`
   - CNN 下采样并提取局部视觉特征，Transformer 对二维 token 序列进行全局关系建模。
   - 使用 ONNX 友好的基础算子（Conv、MatMul、Softmax、LayerNorm、GELU、Reshape/Transpose），避免依赖自定义 CUDA 算子。
   - 训练输出分类 logits 和归一化 embedding；推理/ONNX 输出 embedding。

3. `train.py`
   - CUDA 为默认且必需的训练设备，启用 AMP 混合精度、梯度缩放和 pinned-memory/non-blocking 传输。
   - 实现交叉熵 + batch-hard triplet loss、学习率 warm-up/衰减、checkpoint 保存与断点续训。
   - 提供可选的 SYSU-MM01 query/gallery 评估及多 trial gallery 协议。

4. `export_onnx.py`
   - 从 checkpoint 载入模型并导出 ONNX（opset 17）。
   - 设置动态 batch 轴、固定图像尺寸，并在安装 `onnx` 时执行模型合法性检查。

5. `requirements.txt` 与 `README_PYTORCH.md`
   - 记录 CUDA 版 PyTorch、ONNX 相关依赖、训练/恢复/评估/导出命令和目录约定。

## 4. 模型数据流 / Model data flow

`RGB/IR image -> shared CNN -> flattened spatial tokens + CLS token -> Transformer blocks -> LayerNorm -> embedding head -> L2 embedding`

训练时将成对 RGB/IR batch 沿 batch 维拼接，共享整个网络；分类头学习身份，triplet loss 显式拉近跨模态同身份特征并分离不同身份特征。

During training, paired RGB/IR batches are concatenated on the batch dimension and passed through the shared network. The classifier learns identity discrimination, while triplet loss pulls together same-identity cross-modal features and separates different identities.

## 5. ONNX 兼容边界 / ONNX compatibility boundary

- 输入张量：`images: float32 [N, 3, 288, 144]`。
- 输出张量：`embeddings: float32 [N, embedding_dim]`。
- 动态维度仅包含 batch；高宽固定，避免位置编码和 token 数在部署端产生歧义。
- 导出图只包含前向推理；采样、增强、loss 和评价逻辑不进入 ONNX。

## 6. 验证清单 / Verification checklist

- Python 静态编译通过。
- 数据扫描、身份映射和配对 sampler 的小规模 smoke test 通过。
- 有 CUDA PyTorch 环境时：单 batch 前向、loss、反向和 AMP 更新通过。
- 导出 ONNX 后通过 `onnx.checker.check_model`；若安装 onnxruntime，再比较 PyTorch 与 ONNX 输出误差。
- 确认旧项目和 SYSU-MM01 数据文件没有被修改。

## 7. 实施顺序 / Implementation order

1. 先完成 dataloader 与采样器。
2. 实现 ONNX 友好的 CNN-Transformer。
3. 接入 CUDA 训练、损失、checkpoint 与评估。
4. 增加 ONNX 导出入口和使用说明。
5. 依次执行静态、数据、CUDA 和 ONNX 验证；环境缺少依赖时明确记录未执行项。
