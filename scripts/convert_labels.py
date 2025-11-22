import sys
import os
import glob
import rasterio
import geopandas as gpd
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import LOCAL_DIR, LOCAL_LABELS_DIR
from src.utils import detect_black_bars

# Images are in LOCAL_DIR (flat structure as per data_loader)
IMAGE_DIR = LOCAL_DIR 
LABEL_DIR = LOCAL_LABELS_DIR
OUTPUT_DIR = os.path.join(LOCAL_DIR, "labels_yolo")

def convert_labels():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all GeoJSON files
    geojson_files = glob.glob(os.path.join(LABEL_DIR, "*.geojson"))
    
    print(f"Found {len(geojson_files)} label files in {LABEL_DIR}.")
    print(f"Looking for images in {IMAGE_DIR}.")
    
    converted_count = 0
    
    for geojson_path in tqdm(geojson_files, desc="Converting labels"):
        filename = os.path.basename(geojson_path)
        # SN6_Train_AOI_11_Rotterdam_Buildings_... -> SN6_Train_AOI_11_Rotterdam_PS-RGB_...
        image_filename = filename.replace("Buildings", "PS-RGB").replace(".geojson", ".tif")
        image_path = os.path.join(IMAGE_DIR, image_filename)
        
        if not os.path.exists(image_path):
            # Images might not be downloaded yet or path is wrong
            continue
            
        try:
            with rasterio.open(image_path) as src:
                img_height, img_width = src.height, src.width
                src_transform = src.transform
                src_crs = src.crs
                
                # Detect black bars
                image_array = src.read()
                # Convert from (C, H, W) to (H, W, C) for detection
                if len(image_array.shape) == 3:
                    image_array = np.transpose(image_array, (1, 2, 0))
                
                top_crop, bottom_crop, left_crop, right_crop = detect_black_bars(
                    image_array, threshold=10, min_bar_size=5, bar_threshold=0.95
                )
                
                # Adjust image dimensions for cropping
                effective_height = img_height - top_crop - bottom_crop
                effective_width = img_width - left_crop - right_crop
                
            gdf = gpd.read_file(geojson_path)
            
            yolo_lines = []
            
            if not gdf.empty:
                # Ensure gdf is in the same CRS as image
                # GeoJSON is 4326. Image might be different.
                if src_crs and src_crs.to_string() != "EPSG:4326":
                     try:
                        gdf = gdf.to_crs(src_crs)
                     except Exception:
                        pass # Keep as is if transformation fails, or handle error

                for _, row in gdf.iterrows():
                    geom = row.geometry
                    if geom.is_empty:
                        continue
                    
                    # Get bounds in map units
                    minx, miny, maxx, maxy = geom.bounds
                    
                    # ~src_transform * (x, y) returns (col, row) which is (x, y) in pixels.
                    c1, r1 = ~src_transform * (minx, maxy)
                    c2, r2 = ~src_transform * (maxx, miny)
                    
                    xmin_px = min(c1, c2)
                    xmax_px = max(c1, c2)
                    ymin_px = min(r1, r2)
                    ymax_px = max(r1, r2)
                    
                    # Adjust for black bar cropping
                    xmin_px = xmin_px - left_crop
                    xmax_px = xmax_px - left_crop
                    ymin_px = ymin_px - top_crop
                    ymax_px = ymax_px - top_crop
                    
                    # Clip to cropped image bounds
                    xmin_px = max(0, xmin_px)
                    ymin_px = max(0, ymin_px)
                    xmax_px = min(effective_width, xmax_px)
                    ymax_px = min(effective_height, ymax_px)
                    
                    if xmax_px <= xmin_px or ymax_px <= ymin_px:
                        continue
                    
                    # Convert to YOLO format (normalize by cropped image dimensions)
                    bbox_width = xmax_px - xmin_px
                    bbox_height = ymax_px - ymin_px
                    x_center = xmin_px + bbox_width / 2
                    y_center = ymin_px + bbox_height / 2
                    
                    x_center /= effective_width
                    y_center /= effective_height
                    bbox_width /= effective_width
                    bbox_height /= effective_height
                    
                    # Class 0 for building
                    yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
            
            # Write file even if empty (images with no buildings are valid negatives)
            with open(os.path.join(OUTPUT_DIR, image_filename.replace(".tif", ".txt")), "w") as f:
                f.write("\n".join(yolo_lines))
            
            converted_count += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Converted {converted_count} labels.")

if __name__ == "__main__":
    convert_labels()
