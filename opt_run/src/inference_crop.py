import os
import time
import rasterio
import numpy as np
from PIL import Image
from shapely.geometry import box
from src.vendor_sahi.get_sliced_prediction_fast import get_sliced_prediction_fast
from tqdm import tqdm
from .config import SLICE_HEIGHT, SLICE_WIDTH, OVERLAP_HEIGHT_RATIO, VERBOSE
from time import perf_counter


def run_inference_crop(image_paths, detection_model, analytics_callbacks=None):
    """
    Runs inference on a list of image paths, cropping out nodata/black borders
    before passing the image to SAHI, but keeping georeferencing correct.

    Additionally, measures time spent in each major step of the pipeline
    and aggregates timing reported by SAHI (slice/prediction/postprocess).

    Returns:
        tuple: (geo_detections, inference_cache)
    """
    if analytics_callbacks is None:
        analytics_callbacks = []

    geo_detections = []
    inference_cache = {}

    # timing buckets (seconds) for *your* pipeline
    timings = {
        "read_image": 0.0,
        "build_mask": 0.0,
        "cropping": 0.0,
        "normalization": 0.0,
        "sahi_inference": 0.0,
        "georeferencing": 0.0,
        "callbacks": 0.0,
    }

    # timing buckets reported by SAHI for the model itself
    sahi_timings = {
        "slice": 0.0,
        "prediction": 0.0,
        "postprocess": 0.0,
    }

    def log_time(name, start_ts):
        """Accumulate elapsed time in the timings dict."""
        elapsed = time.perf_counter() - start_ts
        timings[name] = timings.get(name, 0.0) + elapsed

    print(f"Starting inference cropped on {len(image_paths)} images...")
    overall_start = time.perf_counter()

    for path in tqdm(image_paths, desc="Detecting"):
        # --- Open and read full image ---
        t_stage = time.perf_counter()
        with rasterio.open(path) as src:
            img = src.read([1, 2, 3]).transpose(1, 2, 0)  # (H, W, 3)
            transform = src.transform
            nodata = src.nodata
        log_time("read_image", t_stage)

        # --- Build valid mask (True = real data, False = nodata/black) ---
        t_stage = time.perf_counter()
        if nodata is not None:
            valid_mask = ~(
                (img[..., 0] == nodata) &
                (img[..., 1] == nodata) &
                (img[..., 2] == nodata)
            )
        else:
            valid_mask = (img.sum(axis=2) > 0)
        log_time("build_mask", t_stage)

        if not valid_mask.any():
            print(f"Warning: no valid pixels in {path}, skipping.")
            empty_result = None
            inference_cache[path] = empty_result

            # still call callbacks if you want
            t_stage = time.perf_counter()
            for callback in analytics_callbacks:
                try:
                    callback(path, empty_result, img, transform)
                except Exception as e:
                    print(f"Error in analytics callback {callback.__name__}: {e}")
            log_time("callbacks", t_stage)
            continue

        # --- Compute tight bounding box of valid region ---
        t_stage = time.perf_counter()
        ys, xs = np.where(valid_mask)
        y_min, y_max = ys.min(), ys.max() + 1  # +1 because slicing is exclusive
        x_min, x_max = xs.min(), xs.max() + 1
        cropped = img[y_min:y_max, x_min:x_max]
        log_time("cropping", t_stage)

        # --- Normalize cropped image (same logic as before) ---
        t_stage = time.perf_counter()
        p2, p98 = np.percentile(cropped, (2, 98))
        div = (p98 - p2) if (p98 - p2) > 0 else 1
        cropped_norm = np.clip((cropped - p2) / div * 255.0, 0, 255).astype(np.uint8)
        cropped_contiguous = np.ascontiguousarray(cropped_norm)
        log_time("normalization", t_stage)

        # --- Predict on cropped image (SAHI) ---
        t_stage = time.perf_counter()
        result = get_sliced_prediction_fast(
            Image.fromarray(cropped_contiguous),
            detection_model,
            slice_height=SLICE_HEIGHT,
            slice_width=SLICE_WIDTH,
            overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
            overlap_width_ratio=OVERLAP_HEIGHT_RATIO,
            verbose=VERBOSE,
        )
        inference_cache[path] = result
        log_time("sahi_inference", t_stage)

        # --- Accumulate SAHI internal timings for this image ---
        if hasattr(result, "durations_in_seconds") and isinstance(result.durations_in_seconds, dict):
            d = result.durations_in_seconds
            for k in sahi_timings.keys():
                sahi_timings[k] += d.get(k, 0.0)

            # Optional: per-image debug print
            # print(f"{os.path.basename(path)} -> SAHI slice={d.get('slice',0):.3f}s, "
            #       f"pred={d.get('prediction',0):.3f}s, post={d.get('postprocess',0):.3f}s")

        # --- Georeference results (shift back into full-image coords first) ---
        t_stage = time.perf_counter()
        for det in result.object_prediction_list:
            # bbox coords are relative to cropped image; offset them
            x1_full = det.bbox.minx + x_min
            y1_full = det.bbox.miny + y_min
            x2_full = det.bbox.maxx + x_min
            y2_full = det.bbox.maxy + y_min

            # pixel -> geo using original transform
            x1_geo, y1_geo = transform * (x1_full, y1_full)
            x2_geo, y2_geo = transform * (x2_full, y2_full)

            geo_detections.append({
                "geometry": box(
                    min(x1_geo, x2_geo),
                    min(y1_geo, y2_geo),
                    max(x1_geo, x2_geo),
                    max(y1_geo, y2_geo),
                ),
                "label": det.category.name,
                "score": round(det.score.value, 2),
                "file": os.path.basename(path),
            })
        log_time("georeferencing", t_stage)

        # --- Run analytics callbacks on the full image (unchanged API) ---
        t_stage = time.perf_counter()
        for callback in analytics_callbacks:
            try:
                callback(path, result, img, transform)
            except Exception as e:
                print(f"Error in analytics callback {callback.__name__}: {e}")
        log_time("callbacks", t_stage)

    overall_end = time.perf_counter()
    total_inference_time = overall_end - overall_start

    print(f"Done! Saved {len(geo_detections)} detections.")
    print(f"Total inference time (wall-clock): {total_inference_time:.2f} seconds\n")

    # --- Timing summary: your pipeline ---
    print("===== Inference Timing Summary (pipeline) =====")
    for k, v in timings.items():
        print(f"{k:20s}: {v:.4f} seconds")
    print(f"{'TOTAL (sum of stages)':20s}: {sum(timings.values()):.4f} seconds\n")

    # --- Timing summary: SAHI internals ---
    print("===== SAHI Timing Summary (inside get_sliced_prediction) =====")
    for k, v in sahi_timings.items():
        print(f"{k:20s}: {v:.4f} seconds")
    print(f"{'TOTAL (SAHI slice+pred+post)':20s}: {sum(sahi_timings.values()):.4f} seconds")

    return geo_detections, inference_cache
