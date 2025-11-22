# src/inference_openvino.py

import time
import numpy as np
import torch
from openvino import Core

from .inference import run_inference  # reuse the SAHI pipeline


def _patch_detection_model_with_openvino(detection_model, onnx_path: str, device: str = "CPU"):
    core = Core()
    ov_model = core.read_model(onnx_path)
    compiled_model = core.compile_model(ov_model, device)

    # Underlying Ultralytics YOLO model used by SAHI
    yolo_model = detection_model.model

    def openvino_infer(x, *args, **kwargs):
        if isinstance(x, torch.Tensor):
            np_input = x.detach().cpu().numpy().astype(np.float32)
        else:
            np_input = np.asarray(x, dtype=np.float32)

        ov_output = compiled_model([np_input])[0]  # numpy array
        return torch.from_numpy(ov_output)

    # 🔑 Ensure predictor exists before patching
    predictor = getattr(yolo_model, "predictor", None)
    if predictor is None:
        if hasattr(yolo_model, "_new_predictor"):
            # Create a default predictor instance
            yolo_model.predictor = yolo_model._new_predictor(overrides={})
        else:
            raise RuntimeError(
                "Underlying YOLO model has no 'predictor' or '_new_predictor'. "
                "Check what load_model() returns."
            )

    # Now predictor is guaranteed not to be None
    yolo_model.predictor.inference = openvino_infer

    return detection_model


def run_inference_openvino(
    image_paths,
    detection_model,
    onnx_path: str,
    analytics_callbacks=None,
    device: str = "CPU",
):
    if analytics_callbacks is None:
        analytics_callbacks = []

    print(f"Starting OpenVINO+SAHI inference on {len(image_paths)} images...")

    detection_model = _patch_detection_model_with_openvino(
        detection_model=detection_model,
        onnx_path=onnx_path,
        device=device,
    )

    t0 = time.time()
    geo_detections, inference_cache = run_inference(
        image_paths=image_paths,
        detection_model=detection_model,
        analytics_callbacks=analytics_callbacks,
    )
    total = time.time() - t0

    print(f"Total OpenVINO+SAHI inference time: {total:.2f} seconds")
    return geo_detections, inference_cache
