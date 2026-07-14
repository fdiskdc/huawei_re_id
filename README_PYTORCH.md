# SYSU-MM01 PyTorch CNN-Transformer

这是一套位于项目根目录、独立于旧 MindSpore 代码的 RGB/IR 行人重识别实现。数据代码直接读取 `SYSU-MM01/cam*` 下的 JPG，不会改写数据集。

This is a root-level RGB/IR person ReID implementation independent of the legacy MindSpore code. It reads JPG files directly from `SYSU-MM01/cam*` and never rewrites the dataset.

## 文件 / Files

- `data_loader.py`：身份级 train/test/val 划分、P×K 双模态配对采样、query/gallery 协议。
- `model.py`：共享 CNN + ONNX 友好 Transformer + ReID embedding/classifier。
- `train.py`：CUDA AMP、交叉熵 + batch-hard triplet、评估、checkpoint。
- `export_onnx.py`：opset 17、动态 batch 的 embedding 模型导出与校验。
- `md/pytorch_cnn_transformer_plan.md`：实施计划和验证清单。
- `md/pytorch_validation_report.md`：本次静态检查、数据协议检查及未执行项。

## 环境 / Environment

在 WSL2 中进入项目：

```bash
cd /home/dc/vscode/re_id
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

先根据本机 CUDA/驱动选择 PyTorch 官方提供的 CUDA wheel，再安装其余依赖。例如当所选 PyTorch 版本提供 `cu128` wheel 时：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

安装后确认 CUDA：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> CUDA wheel 地址会随 PyTorch 版本变化；若 `cu128` 不适用，请在 PyTorch 官方安装器中选择 Linux、Pip、Python 和与驱动匹配的 CUDA 版本。

## 训练 / Training

默认配置使用 GPU 0、`288×144` 输入、每批 8 个身份、每身份每模态 4 张图。网络实际输入 batch 为 `2×8×4=64`。

```bash
python train.py \
  --data-root ./SYSU-MM01 \
  --output-dir ./outputs \
  --device cuda \
  --gpu 0 \
  --epochs 60 \
  --batch-size 8 \
  --num-pos 4
```

### 7:2:1 划分与早停 / Split and early stopping

默认从 `exp/available_id.txt` 读取可用身份，并按身份执行 `train:test:val = 7:2:1` 的确定性划分。当前数据集 491 个双模态身份会划分为 344/98/49 个身份。同一身份不会同时出现在多个集合中。

- train：参与分类与 triplet loss 训练。
- val：每轮执行 IR query → RGB gallery 评估，用于选择 `best.pt` 和早停。
- test：训练停止后加载 `best.pt`，只做一次最终评估；也供 `--test-only` 使用。

默认监控 val mAP，连续 10 次验证没有至少 `1e-4` 的提升时停止：

```bash
python train.py \
  --split-ratio 7:2:1 \
  --split-seed 0 \
  --eval-every 1 \
  --early-stopping-metric mAP \
  --early-stopping-patience 10 \
  --early-stopping-min-delta 0.0001
```

可将监控指标改成 `rank1` 或 `mINP`。如需关闭停止动作但仍在 val 上选最佳模型，可传入 `--disable-early-stopping`。patience 按“验证次数”计数；若修改 `--eval-every`，对应的 epoch 等待长度也会变化。

checkpoint 会保存实际 train/test/val 身份列表和完整早停状态。断点续训会复用这些身份，不会重新随机划分。

3070 Ti Laptop GPU 只有 8 GB 显存；若显存不足，优先将 `--batch-size` 调到 4 或把 `--transformer-dim`/`--embedding-dim` 调到 192，并保证 `--num-heads` 能整除 transformer dim。

断点续训 / Resume:

```bash
python train.py --resume ./outputs/last.pt --epochs 60
```

只评估 checkpoint / Evaluate only:

```bash
python train.py --resume ./outputs/best.pt --test-only --gallery-trials 10
```

训练默认要求 CUDA，并开启 AMP。只有做轻量代码调试时才建议显式传入 `--device cpu`；正式训练不要使用 CPU。

## ONNX 导出 / ONNX export

```bash
python export_onnx.py \
  --checkpoint ./outputs/best.pt \
  --output ./outputs/cnn_transformer_reid.onnx \
  --opset 17
```

安装 `onnxruntime` 后可额外比较 PyTorch 与 ONNX 输出：

```bash
pip install onnxruntime
python export_onnx.py \
  --checkpoint ./outputs/best.pt \
  --output ./outputs/cnn_transformer_reid.onnx \
  --verify-runtime
```

部署接口固定为：

- 输入 `images`: `float32 [N, 3, 288, 144]`，ImageNet mean/std 标准化。
- 输出 `embeddings`: `float32 [N, 256]`，已做 L2 normalization。
- batch `N` 为动态维；图像高宽固定。若训练时修改 `--img-height`、`--img-width` 或 `--embedding-dim`，导出模型会自动使用 checkpoint 中保存的对应值。
