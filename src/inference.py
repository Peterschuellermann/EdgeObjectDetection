import os
import time
import rasterio
import numpy as np
from PIL import Image
from shapely.geometry import box
from rasterio.warp import transform as transform_coords
from sahi.predict import get_sliced_prediction
from sahi.annotation import BoundingBox
from tqdm import tqdm
from .config import SLICE_HEIGHT, SLICE_WIDTH, OVERLAP_HEIGHT_RATIO, VERBOSE, DST_CRS
from .utils import detect_black_bars

def run_inference(image_paths, detection_model, analytics_callbacks=None, inference_target_size=None):
    """
    Runs inference on a list of image paths.
    
    Args:
        image_paths (list): List of file paths to images.
        detection_model: Loaded SAHI detection model.
        analytics_callbacks (list, optional): List of functions to call with detection results.
                                              Each callback should accept (image_path, result, img_array, transform).
        inference_target_size (tuple, optional): Size to resize image to before inference (width, height).
                                                 Defaults to (SLICE_WIDTH, SLICE_HEIGHT).
    
    Returns:
        tuple: (geo_detections, inference_cache)
    """
    if analytics_callbacks is None:
        analytics_callbacks = []

    if inference_target_size is None:
        inference_target_size = (SLICE_WIDTH, SLICE_HEIGHT)

    geo_detections = []
    inference_cache = {}

    print(f"Starting inference on {len(image_paths)} images...")
    
    start_time = time.time()

    for path in tqdm(image_paths, desc="Detecting"):
        # Open and Read
        with rasterio.open(path) as src:
            img = src.read([1, 2, 3]).transpose(1, 2, 0)
            orig_h, orig_w = img.shape[:2]
            src_crs = src.crs
            original_transform = src.transform
            
            # Detect and crop black bars (same as training)
            top_crop, bottom_crop, left_crop, right_crop = detect_black_bars(
                img, threshold=10, min_bar_size=5, bar_threshold=0.95
            )
            
            # Crop the image
            effective_h = orig_h - top_crop - bottom_crop
            effective_w = orig_w - left_crop - right_crop
            img = img[top_crop:orig_h-bottom_crop, left_crop:orig_w-right_crop, :]
            
            # Adjust transform to account for crop offset
            transform = rasterio.Affine(
                original_transform.a, original_transform.b,
                original_transform.c + left_crop * original_transform.a,
                original_transform.d, original_transform.e,
                original_transform.f + top_crop * original_transform.e
            )

        # Normalize
        p2, p98 = np.percentile(img, (2, 98))
        img = np.clip((img - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
        img_contiguous = np.ascontiguousarray(img)

        # Resize
        pil_img = Image.fromarray(img_contiguous)
        pil_img_resized = pil_img.resize(inference_target_size)
        
        # Use cropped dimensions for scaling
        scale_x = effective_w / inference_target_size[0]
        scale_y = effective_h / inference_target_size[1]

        # Predict
        result = get_sliced_prediction(
            pil_img_resized, 
            detection_model,
            slice_height=SLICE_HEIGHT, 
            slice_width=SLICE_WIDTH, 
            overlap_height_ratio=OVERLAP_HEIGHT_RATIO, 
            verbose=VERBOSE
        )
        
        # Scale back detections to cropped image size
        for det in result.object_prediction_list:
            box_obj = det.bbox
            # BoundingBox expects [minx, miny, maxx, maxy]
            new_box = [
                box_obj.minx * scale_x,
                box_obj.miny * scale_y,
                box_obj.maxx * scale_x,
                box_obj.maxy * scale_y
            ]
            det.bbox = BoundingBox(new_box)

        inference_cache[path] = result

        # Georeference results
        for det in result.object_prediction_list:
            x1, y1 = transform * (det.bbox.minx, det.bbox.miny)
            x2, y2 = transform * (det.bbox.maxx, det.bbox.maxy)

            # Calculate centroid in source CRS
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Transform to lat/lon (EPSG:4326)
            # Try with always_xy=True (rasterio >= 1.3.0), fall back to old API if not supported
            try:
                lon, lat = transform_coords(src_crs, DST_CRS, [cx], [cy], always_xy=True)
            except TypeError:
                # Older rasterio version - transform returns (lat, lon) for EPSG:4326, so swap
                lat, lon = transform_coords(src_crs, DST_CRS, [cy], [cx])

            geo_detections.append({
                'geometry': box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
                'label': det.category.name,
                'score': round(det.score.value, 2),
                'file': os.path.basename(path),
                'latitude': lat[0],
                'longitude': lon[0]
            })
        
        # Run analytics callbacks
        for callback in analytics_callbacks:
            try:
                callback(path, result, img, transform)
            except Exception as e:
                print(f"Error in analytics callback {callback.__name__}: {e}")

    end_time = time.time()
    total_inference_time = end_time - start_time

    print(f"Done! Saved {len(geo_detections)} detections.")
    print(f"Total inference time: {total_inference_time:.2f} seconds")
    
    return geo_detections, inference_cache
