import sys, os
sys.path.append(os.path.abspath("."))

from src.export_openvino import export_dior_yolo_to_onnx
from openvino import Core

# 1. Export PyTorch model to ONNX
onnx_path = export_dior_yolo_to_onnx(imgsz=640)

# 2. Load and compile with OpenVINO
core = Core()
ov_model = core.read_model(onnx_path)
compiled_model = core.compile_model(ov_model, "CPU")

print("OpenVINO model compiled successfully")
