# 早停与 7:2:1 数据划分修改计划

## 目标 / Goal

- 按身份而非图片将可用 SYSU-MM01 身份划分为 train:test:val = 7:2:1，防止同一身份跨集合造成数据泄漏。
- 每轮使用 val 集的 IR query 与 RGB gallery 计算 ReID 指标。
- 默认监控 val mAP；连续若干次没有达到最小提升时提前停止。
- 训练结束后恢复最佳 checkpoint，仅在独立 test 集上报告最终指标。

## 修改项 / Changes

1. `data_loader.py`
   - 增加确定性的身份级比例划分函数。
   - 允许训练、query 和 gallery 构建函数接收显式身份列表。

2. `train.py`
   - 增加 `--split-ratio 7:2:1`、`--split-seed`。
   - 增加 `--early-stopping-patience`、`--early-stopping-min-delta` 和 `--early-stopping-metric`。
   - checkpoint 保存实际身份划分和早停状态，保证恢复训练时不重新随机划分。
   - val 用于选模/早停，test 只用于最终评估或 `--test-only`。

3. `README_PYTORCH.md`
   - 增加划分、早停语义及命令示例。

## 验证 / Verification

- 静态编译与 AST 检查。
- 检查三组身份互斥、并集完整、数量符合 7:2:1 的最大余数分配。
- 检查每组身份均具有 RGB 和 IR 图像。
- 遵照用户要求，不安装运行环境；CUDA/ONNX 动态测试不在本次执行。
