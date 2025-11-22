import os
from pathlib import Path
from botocore import UNSIGNED
from botocore.config import Config

# S3 Configuration
CLIENT_CONFIG = Config(signature_version=UNSIGNED, max_pool_connections=20)
BUCKET = 'spacenet-dataset'
PREFIX = 'spacenet/SN6_buildings/train/AOI_11_Rotterdam/PS-RGB/'
LOCAL_DIR = "/Volumes/aerospace/rotterdam_opt"


# Model Configuration
MODEL_REPO_ID = "pauhidalgoo/yolov8-DIOR"
MODEL_FILENAME = "DIOR_yolov8n_backbone.pt"
MODEL_CONFIDENCE_THRESHOLD = 0.4
MODEL_DEVICE = "cpu"

# Inference Configuration
SLICE_HEIGHT = 1024
SLICE_WIDTH = 1024
OVERLAP_HEIGHT_RATIO = 0.1
VERBOSE = 0

# Visualization
DST_CRS = 'EPSG:4326'

