import os
from pathlib import Path
from botocore import UNSIGNED
from botocore.config import Config

# S3 Configuration
CLIENT_CONFIG = Config(signature_version=UNSIGNED, max_pool_connections=20)
BUCKET = 'spacenet-dataset'
PREFIX = 'spacenet/SN6_buildings/train/AOI_11_Rotterdam/PS-RGB/'
LOCAL_DIR = os.path.join(str(Path.home()), "rotterdam_opt")

import torch

# Model Configuration
MODEL_REPO_ID = "pauhidalgoo/yolov8-DIOR"
MODEL_FILENAME = "DIOR_yolov8n_backbone.pt"
MODEL_CONFIDENCE_THRESHOLD = 0.4

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

MODEL_DEVICE = get_device()
print(f"Inference device selected: {MODEL_DEVICE}")

# Inference Configuration
SLICE_HEIGHT = 512
SLICE_WIDTH = 512
OVERLAP_HEIGHT_RATIO = 0.1
VERBOSE = 0

# Visualization
DST_CRS = 'EPSG:4326'
MAPS_OUTPUT_DIR = "maps"

# AIS Configuration
AIS_PROVIDER = "local_csv"  # Options: "datalastic", "local_csv"
# Read from environment variable, or use hardcoded value as fallback
AIS_API_KEY = os.getenv("AIS_API_KEY", "9aac3244-93b4-491e-b7b0-b0bcd70df0de")
ROTTERDAM_BBOX = {
    "min_lon": 4.0,
    "max_lon": 4.5,
    "min_lat": 51.85,
    "max_lat": 51.95
}
AIS_MATCH_DISTANCE_M = 100  # meters - distance threshold for matching
AIS_TIME_BUFFER_SEC = 300   # ±5 minutes buffer around image time range
AIS_LOCAL_CSV_PATH = "ais_data.csv"  # Path to local CSV file if using local_csv provider

