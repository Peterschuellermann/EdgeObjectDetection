# src/export_openvino.py
import os
os.environ["ULTRALYTICS_AUTO_INSTALL"] = "0"

from huggingface_hub import hf_hub_download
import torch

MODEL_REPO_ID = "pauhidalgoo/yolov8-DIOR"
WEIGHTS_FILENAME = "DIOR_yolov8n_backbone.pt"


def export_dior_yolo_to_onnx(imgsz: int = 640,
                             onnx_path: str = "dior_yolov8n_obb.onnx") -> str:
    weights_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=WEIGHTS_FILENAME,
    )
    print(f"Downloaded weights to: {weights_path}")

    # Make sure ultralytics is importable for the pickled classes
    import ultralytics  # noqa: F401

    ckpt = torch.load(weights_path, map_location="cpu")

    # Prefer EMA weights if present
    raw_model = ckpt.get("ema") or ckpt["model"]

    # 🔑 Ensure model is on CPU and in float32, not half
    raw_model.to("cpu")
    raw_model.float()          # <--- this is the crucial line
    raw_model.eval()

    # Dummy input in float32 (default)
    dummy_input = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32)

    onnx_path = os.path.abspath(onnx_path)
    with torch.no_grad():      # optional but nice
        torch.onnx.export(
            raw_model,
            dummy_input,
            onnx_path,
            opset_version=12,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={
                "images": {0: "batch"},
                "output": {0: "batch"},
            },
        )

    print(f"Exported ONNX model to: {onnx_path}")
    return onnx_path
