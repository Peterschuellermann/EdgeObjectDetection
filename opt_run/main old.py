USE_OPENVINO = False  # set False to use pure SAHI / PyTorch

import glob
import os

from src.utils import setup_logging, inspect_environment
from src.data_loader import list_s3_files, download_files_parallel
from src.model_handler import load_model
from src.inference import run_inference
from src.inference_crop import run_inference_crop  # still here if you use it
from src.inference_openvino import run_inference_openvino

from src.visualization import plot_results, create_map
from src.config import LOCAL_DIR


def main():
    # 1. Setup
    setup_logging()
    inspect_environment()

    # 2. Data Download
    tasks = list_s3_files()
    download_files_parallel(tasks)

    # 3. Load SAHI/Ultralytics model (used in both modes)
    print("Loading model...")
    detection_model = load_model()

    # 4. Select subset of files
    folder = LOCAL_DIR
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))[:20]

    # 5. Run Inference (backend switch)
    if USE_OPENVINO:
        print("Running OpenVINO+SAHI pipeline...")
        geo_detections, inference_cache = run_inference_openvino(
            files,
            detection_model=detection_model,
            onnx_path="dior_yolov8n_obb.onnx",  # adjust path if needed
        )
    else:
        print("Running SAHI/PyTorch pipeline...")
        geo_detections, inference_cache = run_inference(
            files,
            detection_model,
        )

    # 6. Visualization (shared)
    print("Generating plots...")
    plot_results(files, inference_cache)

    print("Generating map...")
    create_map(files, geo_detections)


if __name__ == "__main__":
    main()
