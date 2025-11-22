import sys, os
sys.path.append(os.path.abspath("."))

from src.export_openvino import export_dior_yolo_to_onnx
from openvino import Core
import numpy as np
import time

def compile_openvino_model(onnx_path):
    core = Core()
    model = core.read_model(onnx_path)
    return core.compile_model(model, "CPU")

def openvino_infer_fn(compiled_model):
    def infer_fn(batch):
        return compiled_model([batch])[0]
    return infer_fn

def benchmark(fn, batch, runs=100):
    # warmup
    for _ in range(10):
        fn(batch)

    t0 = time.time()
    for _ in range(runs):
        fn(batch)
    t1 = time.time()

    avg_time = (t1 - t0) / runs
    print(f"Average time per run: {avg_time*1000:.2f} ms")
    print(f"Throughput: {1/avg_time:.2f} FPS")

if __name__ == "__main__":
    onnx_path = export_dior_yolo_to_onnx(imgsz=640)
    compiled = compile_openvino_model(onnx_path)
    infer_fn = openvino_infer_fn(compiled)

    batch = np.zeros((1, 3, 640, 640), dtype=np.float32)
    benchmark(infer_fn, batch)
