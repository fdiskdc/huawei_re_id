"""Export a trained checkpoint to ONNX / 将训练 checkpoint 导出为 ONNX。"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import torch

from model import CNNTransformerReID, EmbeddingOnly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CNN-Transformer ReID to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/cnn_transformer_reid.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--verify-runtime", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found / checkpoint 不存在: {checkpoint_path}")
    if args.opset < 17:
        raise ValueError("This model requires ONNX opset >= 17 / 本模型要求 ONNX opset >= 17")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested but CUDA is unavailable")

    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_config" not in checkpoint or "model" not in checkpoint:
        raise KeyError("Checkpoint must contain model_config and model")

    model = CNNTransformerReID(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    export_model = EmbeddingOnly(model).eval().to(device)
    config = checkpoint["model_config"]
    example = torch.randn(
        1,
        3,
        int(config["image_height"]),
        int(config["image_width"]),
        device=device,
    )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Only batch is dynamic; fixed H/W keeps positional tokens deterministic.
    # 仅 batch 为动态维；固定高宽保证位置 token 确定。
    export_options = {
        "input_names": ["images"],
        "output_names": ["embeddings"],
        "dynamic_axes": {"images": {0: "batch"}, "embeddings": {0: "batch"}},
        "opset_version": args.opset,
        "do_constant_folding": True,
    }
    # PyTorch 2.9 defaults to the dynamo exporter, which needs extra packages.
    # 新版 PyTorch 默认 dynamo exporter；这里显式使用兼容性更广的传统导出器。
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_options["dynamo"] = False
    torch.onnx.export(export_model, example, output_path, **export_options)

    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("Install onnx to validate the exported graph / 请安装 onnx 后校验") from error
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed / ONNX 校验通过: {output_path}")

    if args.verify_runtime:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("--verify-runtime requires onnxruntime or onnxruntime-gpu") from error
        with torch.inference_mode():
            expected = export_model(example).float().cpu().numpy()
        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        actual = session.run(["embeddings"], {"images": example.float().cpu().numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
        print("ONNX Runtime output matches PyTorch / ONNX Runtime 输出与 PyTorch 一致")


if __name__ == "__main__":
    main()
