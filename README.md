# Edge Object Detection

This project provides an "edge-ready" workflow for performing object detection on high-resolution satellite imagery using limited hardware resources (CPU only). It targets the detection of objects in large satellite images of the Port of Rotterdam using a lightweight YOLOv8 model and slicing techniques.

## Project Structure

- `main.py`: Entry point script that runs the full pipeline (Environment Check -> Download -> Inference -> AIS Matching -> Visualization).
- `dashboard.py`: Streamlit dashboard for visualizing detection analytics and statistics.
- `Edge_Ready_Object_Detection_Demo_vFinal.ipynb`: A Jupyter notebook demonstrating the workflow.
- `colab_training.ipynb`: A notebook for retraining the model on Google Colab/cloud GPUs.
- `src/`: Source code modules.
    - `config.py`: Configuration constants (S3 paths, model settings, AIS configuration).
    - `data_loader.py`: S3 file listing and parallel downloading.
    - `model_handler.py`: Model downloading and initialization.
    - `inference.py`: Core inference logic with support for analytics callbacks and black bar cropping.
    - `visualization.py`: Plotting and interactive map generation functions.
    - `utils.py`: Helper functions for environment inspection, black bar detection, and filename parsing.
    - `database.py`: SQLite database operations for storing detections with AIS match information.
    - `ais_workflow.py`: AIS matching workflow integration.
    - `analytics/`: Directory for custom analytics modules.
        - `recorder.py`: Detection recorder that saves results to the database.
    - `ais/`: AIS (Automatic Identification System) integration modules.
        - `provider.py`: AIS data providers (Datalastic API, local CSV).
        - `matcher.py`: Spatial-temporal matching of detections with AIS positions.
- `scripts/`: Data preparation and training utilities.
    - `convert_labels.py`: Convert SpaceNet labels to YOLO format with black bar handling.
    - `prepare_data.py`: Data preparation utilities.
    - `setup_yolo_dataset.py`: YOLO dataset setup scripts.
- `test_integration.py`: Integration tests for AIS workflow components.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd EdgeObjectDetection
    ```

2.  **Set up a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure AIS (Optional):**
    If you want to use AIS ship identification, set the `AIS_API_KEY` environment variable:
    ```bash
    export AIS_API_KEY="your_api_key_here"
    ```
    Or configure the AIS provider in `src/config.py`. See the [AIS Integration](#ais-integration) section for details.

## Usage

### Running the Main Script

To run the complete pipeline from the command line:

```bash
python3 main.py
```

This script will:
1.  Check and print environment statistics (CPU, RAM).
2.  Download dataset images from S3 (if not already present in `~/rotterdam_opt`).
3.  Download and load the YOLOv8 model.
4.  Run inference on a subset of images (with automatic black bar cropping).
5.  Save detections to SQLite database (`detections.db`).
6.  Match detections with AIS data for ship identification (if configured).
7.  Generate visualization plots and save an interactive map to `detection_map.html` with AIS match status.

### Retraining the Model (Cloud)

To retrain the model for better performance or different classes, use the provided `colab_training.ipynb` notebook.

**Why run in the cloud?**
Training deep learning models requires significant computational power (GPUs). Local machines (especially those without NVIDIA GPUs or with complex driver setups like MPS on Mac) can be slow or unstable for training.

**How to run:**
1.  Open [Google Colab](https://colab.research.google.com).
2.  Upload `colab_training.ipynb`.
3.  Set the runtime to **GPU** (Runtime > Change runtime type > T4 GPU).
4.  Run all cells. The notebook will:
    - Download the SpaceNet dataset directly to the cloud instance.
    - Convert labels to the correct format.
    - Train a YOLOv8-Nano model.
    - Zip the trained weights for you to download.

After training, you can update `MODEL_REPO_ID` or `MODEL_FILENAME` in `src/config.py` to use your new custom model.

### Data Preparation Scripts

The `scripts/` directory contains utilities for preparing training data:

- `convert_labels.py`: Converts SpaceNet annotation format to YOLO format, automatically detecting and accounting for black bars in images.
- `setup_yolo_dataset.py`: Sets up the YOLO dataset structure.
- `prepare_data.py`: Additional data preparation utilities.

See `BLACK_BARS_INFO.md` for details on black bar detection and handling.

## AIS Integration

The application includes Automatic Identification System (AIS) integration for ship identification. Detected ships can be matched with AIS data to identify vessel names, MMSI numbers, and vessel types.

### Configuration

Configure AIS in `src/config.py`:

- `AIS_PROVIDER`: Choose between `"datalastic"` (API) or `"local_csv"` (local CSV file).
- `AIS_API_KEY`: Set via environment variable or directly in config (for Datalastic provider).
- `AIS_LOCAL_CSV_PATH`: Path to local CSV file (for local CSV provider).
- `AIS_MATCH_DISTANCE_M`: Distance threshold for matching (default: 100 meters).
- `AIS_TIME_BUFFER_SEC`: Time window buffer around image capture time (default: ±5 minutes).

### Usage

AIS matching runs automatically after inference if configured. The results are:
- Stored in the database with AIS match information (MMSI, vessel name, vessel type, distance).
- Visualized on the interactive map (green markers for identified ships, red for unknown).
- Available in the dashboard for analysis.

## Database

The application uses SQLite (`detections.db`) to store all detection results with the following information:
- Detection metadata (run_id, timestamp, file_name, label, score)
- Geographic coordinates (latitude, longitude)
- Geometry (WKT format)
- AIS match information (if matched: MMSI, vessel name, vessel type, distance)

The database schema is automatically initialized and migrated on first run.

## Dashboard

A Streamlit dashboard is available for visualizing detection analytics:

```bash
streamlit run dashboard.py
```

The dashboard provides:
- Overview metrics (total detections, unique images, classes detected)
- Interactive filters (by run ID, label, confidence score)
- Class distribution charts
- Confidence score histograms
- Detailed results table

## Black Bar Detection

The application automatically detects and crops black bars from SpaceNet images during both training and inference. This ensures:
- Consistent preprocessing between training and inference
- Better model performance (focus on actual image content)
- Correct bounding box coordinates and georeferencing

See `BLACK_BARS_INFO.md` for detailed information about black bars and the implementation.

## Custom Analytics

The inference engine supports plugging in custom analytics functions. You can pass a list of callback functions to `run_inference`. Each callback receives the image path, detection results, and image data, allowing you to perform side-effects (like logging, counting, or database updates) as inference proceeds.

Example:
```python
from src.inference import run_inference

def my_analytics(path, result, img, transform):
    print(f"Processed {path}: {len(result.object_prediction_list)} objects detected")

# ... load model ...
run_inference(files, model, analytics_callbacks=[my_analytics])
```

## Testing

Run integration tests to verify AIS workflow components:

```bash
python3 test_integration.py
```

## Evaluation

The project is designed for evaluation on constrained hardware. The key performance metric is inference time on CPU.
