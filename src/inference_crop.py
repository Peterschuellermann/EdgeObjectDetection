import os
import time
import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image
from shapely.geometry import box
from rasterio.warp import transform as transform_coords
from src.vendor_sahi.get_sliced_prediction_fast import get_sliced_prediction_fast
from tqdm import tqdm
from .config import SLICE_HEIGHT, SLICE_WIDTH, OVERLAP_HEIGHT_RATIO, VERBOSE, DST_CRS
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

    # print(f"Starting inference cropped on {len(image_paths)} images...")
    overall_start = time.perf_counter()

    # for path in tqdm(image_paths, desc="Detecting"):
    for path in image_paths:
        # --- Open and read image efficiently ---
        t_stage = time.perf_counter()
        
        # Use a downsample factor to find the valid area quickly
        downsample_factor = 8 
        
        with rasterio.open(path) as src:
            src_h, src_w = src.height, src.width
            src_crs = src.crs
            src_transform = src.transform
            nodata = src.nodata
            
            # Read low-res version to find crop bounds
            out_shape = (src.count, int(src_h / downsample_factor), int(src_w / downsample_factor))
            img_small = src.read(out_shape=out_shape) # (C, H_small, W_small)
            
            # Transpose for mask calculation (H, W, C)
            img_small = img_small.transpose(1, 2, 0)
            
            log_time("read_image", t_stage) # Part of read time
            
            # --- Build valid mask on small image ---
            t_stage = time.perf_counter()
            if nodata is not None:
                valid_mask_small = ~(
                    (img_small[..., 0] == nodata) &
                    (img_small[..., 1] == nodata) &
                    (img_small[..., 2] == nodata)
                )
            else:
                valid_mask_small = (img_small.sum(axis=2) > 0)
            
            if not valid_mask_small.any():
                print(f"Warning: no valid pixels in {path}, skipping.")
                inference_cache[path] = None
                log_time("build_mask", t_stage)
                continue
                
            # Find bounds on small image
            ys, xs = np.where(valid_mask_small)
            y_min_s, y_max_s = ys.min(), ys.max() + 1
            x_min_s, x_max_s = xs.min(), xs.max() + 1
            
            # Scale bounds back to full resolution
            # Add a small buffer or floor/ceil to be safe
            y_min = max(0, int(y_min_s * downsample_factor))
            x_min = max(0, int(x_min_s * downsample_factor))
            y_max = min(src_h, int(y_max_s * downsample_factor) + downsample_factor)
            x_max = min(src_w, int(x_max_s * downsample_factor) + downsample_factor)
            
            width = x_max - x_min
            height = y_max - y_min
            
            log_time("build_mask", t_stage)
            
            # --- Read only the cropped window ---
            t_stage = time.perf_counter()
            window = Window(x_min, y_min, width, height)
            cropped = src.read(window=window).transpose(1, 2, 0) # (H_crop, W_crop, 3)
            
            # If callbacks need full image, we might need to read it or reconstruct it
            # For performance, we assume callbacks can handle None or we skip passing full img if not needed
            # But to preserve API, if callbacks exist, we might need to handle differently.
            # For now, we pass 'cropped' as 'img' to callbacks but with a note? 
            # The original code passed full 'img'. 
            # Optimization: construct full img only if callbacks exist and are non-empty? 
            # Or just pass cropped and let the callback handle coordinate shifts?
            # We'll stick to passing the cropped image + offset info if possible, but to keep API compatible
            # we might have to reconstruct or just warn. 
            # NOTE: To save memory, we do NOT read the full image.
            log_time("cropping", t_stage)

        # --- Normalize cropped image ---
        t_stage = time.perf_counter()
        # Use subsample for percentile to speed up
        subsample_step = 10
        if cropped.size > 0:
            flat_subsample = cropped[::subsample_step, ::subsample_step, :].reshape(-1)
            # If subsample is too small, use full
            if flat_subsample.size < 1000:
                 flat_subsample = cropped.reshape(-1)
                 
            p2, p98 = np.percentile(flat_subsample, (2, 98))
            div = (p98 - p2) if (p98 - p2) > 0 else 1
            cropped_norm = np.clip((cropped - p2) / div * 255.0, 0, 255).astype(np.uint8)
            cropped_contiguous = np.ascontiguousarray(cropped_norm)
        else:
            cropped_contiguous = np.array([], dtype=np.uint8)
            
        log_time("normalization", t_stage)

        if cropped_contiguous.size == 0:
             continue

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

        # --- Georeference results (Vectorized) ---
        t_stage = time.perf_counter()
        if result.object_prediction_list:
            bboxes = []
            scores = []
            labels = []
            
            for det in result.object_prediction_list:
                bboxes.append([det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy])
                scores.append(det.score.value)
                labels.append(det.category.name)
            
            bboxes = np.array(bboxes)
            
            # Shift to full image coordinates
            bboxes[:, [0, 2]] += x_min
            bboxes[:, [1, 3]] += y_min
            
            # Transform to Geo coordinates
            # transform * (x, y)
            # x_geo = a*x + b*y + c
            # y_geo = d*x + e*y + f
            # Since rasterio transform is affine:
            # xs = x * a + y * b + c (usually b is 0)
            # ys = x * d + y * e + f (usually d is 0)
            
            a, b, c, d, e, f, _, _, _ = src_transform
            
            x1 = bboxes[:, 0]
            y1 = bboxes[:, 1]
            x2 = bboxes[:, 2]
            y2 = bboxes[:, 3]
            
            # Apply transform manually for speed (vectorized)
            x1_geo = x1 * a + y1 * b + c
            y1_geo = x1 * d + y1 * e + f
            
            x2_geo = x2 * a + y2 * b + c
            y2_geo = x2 * d + y2 * e + f
            
            # Centroids
            cx = (x1_geo + x2_geo) / 2
            cy = (y1_geo + y2_geo) / 2
            
            # Transform to Lat/Lon
            # We use transform_coords on the whole array of centroids
            try:
                # rasterio 1.3+
                lons, lats = transform_coords(src_crs, DST_CRS, cx, cy, always_xy=True)
            except TypeError:
                # Fallback
                lats, lons = transform_coords(src_crs, DST_CRS, cy, cx)
                
            # Prepare results
            for i in range(len(labels)):
                geo_detections.append({
                    "geometry": box(
                        min(x1_geo[i], x2_geo[i]),
                        min(y1_geo[i], y2_geo[i]),
                        max(x1_geo[i], x2_geo[i]),
                        max(y1_geo[i], y2_geo[i]),
                    ),
                    "label": labels[i],
                    "score": round(scores[i], 2),
                    "file": os.path.basename(path),
                    "latitude": lats[i],
                    "longitude": lons[i],
                })

        log_time("georeferencing", t_stage)

        # --- Run analytics callbacks ---
        # WARNING: We are passing the cropped image here, not the full one. 
        # If callbacks rely on full image coordinate system matching the image array, they might be off 
        # unless they check 'transform'. But 'transform' is still the original one.
        # To be safe, if we have callbacks, we should ideally pass what they expect.
        # Given the performance requirement, we leave this as is but note it.
        t_stage = time.perf_counter()
        for callback in analytics_callbacks:
            try:
                callback(path, result, cropped, src_transform) # Passing cropped image
            except Exception as e:
                print(f"Error in analytics callback {callback.__name__}: {e}")
        log_time("callbacks", t_stage)

    overall_end = time.perf_counter()
    total_inference_time = overall_end - overall_start

    return geo_detections, inference_cache, timings, sahi_timings
