# Edge Object Detection

This project provides an "edge-ready" workflow for performing object detection on high-resolution satellite imagery using limited hardware resources (CPU only). It targets the detection of objects in large satellite images of the Port of Rotterdam using a lightweight YOLOv8 model and slicing techniques.

## Project Structure

- `main.py`: Entry point script that runs the full pipeline (Environment Check -> Download -> Inference -> Visualization).
- `Edge_Ready_Object_Detection_Demo_vFinal.ipynb`: A Jupyter notebook demonstrating the workflow.
- `src/`: Source code modules.
    - `config.py`: Configuration constants (S3 paths, model settings).
    - `data_loader.py`: S3 file listing and parallel downloading.
    - `model_handler.py`: Model downloading and initialization.
    - `inference.py`: Core inference logic with support for analytics callbacks.
    - `visualization.py`: Plotting and map generation functions.
    - `utils.py`: Helper functions for environment inspection.
    - `analytics/`: Directory for custom analytics modules.

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
4.  Run inference on a subset of images.
5.  Generate visualization plots and save an interactive map to `detection_map.html`.

### Running the Notebook

You can also run the interactive Jupyter notebook:

```bash
jupyter notebook Edge_Ready_Object_Detection_Demo_vFinal.ipynb
```

The notebook has been refactored to use the modular code in `src/` but retains the original flow and explanations.

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

## Evaluation

The project is designed for evaluation on constrained hardware. The key performance metric is inference time on CPU.
