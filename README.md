# Edge Object Detection

This project provides an "edge-ready" workflow for performing object detection on high-resolution satellite imagery using limited hardware resources (CPU only). It targets the detection of objects in large satellite images of the Port of Rotterdam using a lightweight YOLOv8 model and slicing techniques.

## Project Structure

- `main.py`: Entry point script that runs the full pipeline (Environment Check -> Download -> Inference -> Visualization).
- `Edge_Ready_Object_Detection_Demo_vFinal.ipynb`: A Jupyter notebook demonstrating the workflow.
- `colab_training.ipynb`: A notebook for retraining the model on Google Colab/cloud GPUs.
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
