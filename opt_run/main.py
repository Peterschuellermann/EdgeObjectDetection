
import glob
import os

from time import perf_counter

from src.utils import setup_logging, inspect_environment
from src.data_loader import list_s3_files, download_files_parallel
from src.model_handler import load_model
from src.inference import run_inference
from src.inference_crop import run_inference_crop  # still here if you use it

from src.visualization import plot_results, create_map
from src.config import LOCAL_DIR

from concurrent.futures import ProcessPoolExecutor, as_completed


def chunked(seq, n):
    """Yield successive chunks of size n from seq."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def process_batch(file_batch):
    """
    Runs run_inference_crop on a subset of files in a separate process.

    Each process loads its own detection model once.
    """
    # Local import to avoid issues when this function is pickled
    from src.model_handler import load_model
    from src.inference_crop import run_inference_crop

    detection_model = load_model()
    geo_detections, inference_cache = run_inference_crop(file_batch, detection_model)
    return geo_detections, inference_cache


def main():
    # 1. Setup
    setup_logging()
    inspect_environment()

    # 2. Data Download
    tasks = list_s3_files()
    download_files_parallel(tasks)



    # 3. (We no longer need to load the model here for the parallel path)
    # print("Loading model...")
    # detection_model = load_model()

    # 4. Select subset of files
    folder = LOCAL_DIR
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))[:100]

    # 5. Run Inference in parallel over chunks of files
    print("Running SAHI/PyTorch pipeline in parallel...")

    t0 = perf_counter()   # <--- START global timer

    num_workers = 16
    if len(files) < num_workers:
        num_workers = len(files)

    chunk_size = max(1, len(files) // num_workers)

    all_geo_detections = []
    all_inference_cache = {}

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = []
        for batch in chunked(files, chunk_size):
            futures.append(ex.submit(process_batch, batch))

        for fut in as_completed(futures):
            geo_dets, cache = fut.result()
            all_geo_detections.extend(geo_dets)
            all_inference_cache.update(cache)

    t1 = perf_counter()   # <--- END global timer

    print(f"\n✅ Total parallel inference wall time: {t1 - t0:.2f} seconds\n")

    geo_detections = all_geo_detections
    inference_cache = all_inference_cache



    # 6. Visualization (shared)
    print("Generating plots...")
    plot_results(files, inference_cache)

    print("Generating map...")
    create_map(files, geo_detections)


if __name__ == "__main__":
    main()
