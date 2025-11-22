import glob
import os
import uuid
from src.utils import setup_logging, inspect_environment
from src.data_loader import list_s3_files, download_files_parallel
from src.model_handler import load_model
from src.inference import run_inference
from src.visualization import plot_results, create_map
from src.config import LOCAL_DIR
from src.database import init_db
from src.analytics.recorder import DetectionRecorder

def main():
    # 1. Setup
    setup_logging()
    inspect_environment()
    
    # Initialize Database
    init_db()
    run_id = str(uuid.uuid4())
    print(f"Starting Run ID: {run_id}")

    # 2. Data Download
    tasks = list_s3_files()
    download_files_parallel(tasks)

    # 3. Load Model
    model = load_model()

    # 4. Run Inference
    # Select a subset of files for demonstration as in the notebook
    folder = LOCAL_DIR
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))[:100]
    
    # Create recorder callback
    recorder = DetectionRecorder(run_id)
    
    geo_detections, inference_cache = run_inference(files, model, analytics_callbacks=[recorder])

    # 5. Visualization
    print("Generating plots...")
    plot_results(files, inference_cache)
    
    print("Generating map...")
    create_map(files, geo_detections, output_file="detection_map.html")

if __name__ == "__main__":
    main()
