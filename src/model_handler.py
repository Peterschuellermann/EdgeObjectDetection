from huggingface_hub import hf_hub_download
from sahi import AutoDetectionModel
from .config import MODEL_REPO_ID, MODEL_FILENAME, MODEL_CONFIDENCE_THRESHOLD, MODEL_DEVICE

def load_model():
    """Downloads and initializes the YOLOv8 model."""
    # print("Loading model...")
    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8', 
        model_path=model_path, 
        confidence_threshold=MODEL_CONFIDENCE_THRESHOLD, 
        device=MODEL_DEVICE
    )
    return detection_model

