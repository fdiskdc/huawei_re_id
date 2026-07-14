# PyTorch CNN-Transformer 验证报告

验证日期：2026-07-14

## 已完成 / Completed

- `data_loader.py`、`model.py`、`train.py`、`export_onnx.py` 均通过 Python AST 解析和 `py_compile` 静态编译。
- 源码未发现 tab、行尾空格、MindSpore import、旧 `/root/vscode` 绝对路径或旧预处理 `.npy` 依赖。
- SYSU-MM01 协议文件解析结果：train 296 个身份、val 99 个身份，合并后 395 个身份均同时具有 RGB 和 IR 图像。
- 原始数据扫描结果：训练 RGB 22,258 张、训练 IR 11,909 张、IR query 3,803 张；all 模式的单个 gallery trial 包含 301 个身份/相机采样槽位。
- 抽样 JPEG 文件存在、非空且可被系统识别；训练预处理会统一 resize 到 `288×144`。
- 已确认 WSL2 可由 `nvidia-smi` 识别 NVIDIA GeForce RTX 3070 Ti Laptop GPU（8 GB）。
- 临时依赖下载进程已停止，`/tmp/reid-validation-venv` 已清理，项目目录内没有创建虚拟环境。

## 未执行 / Not executed

- 当前 WSL Python 未安装 PyTorch、NumPy、Pillow、ONNX；遵照用户“不安装环境”的要求，没有安装任何运行依赖。
- 因此未执行真实 DataLoader batch、CUDA AMP 前向/反向、`torch.onnx.export`、`onnx.checker` 或 ONNX Runtime 数值对比。

这些运行时检查已经在代码和 `README_PYTORCH.md` 中提供对应入口，待用户在已有 PyTorch CUDA 环境中自行运行。

## 7:2:1 与早停增量验证 / Split and early-stopping update

- `available_id.txt` 中共有 491 个同时具备 RGB/IR 的可用身份。
- `7:2:1` 经最大余数分配后为 train 344、test 98、val 49 个身份。
- 文件级检查确认三组交集为空、并集覆盖全部 491 个身份，并且每个身份均存在 RGB/IR 图像。
- train/val/test API、早停状态和 checkpoint 身份划分均通过 AST 与 `py_compile` 静态检查。
- 遵照用户要求，未安装 PyTorch 环境，因此未执行真实 val/test loader 或早停训练循环。
