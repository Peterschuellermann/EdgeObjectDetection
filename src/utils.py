import sys
import os
import psutil
import torch
import logging
import re
import numpy as np
from datetime import datetime

def setup_logging():
    """Configures minimal logging."""
    logging.getLogger("urllib3").setLevel(logging.ERROR)

def inspect_environment():
    """Prints environment statistics."""
    print("--- Inspecting The Environment ---")

    # Check python version
    print("Python Version:")
    print(sys.version)

    # Use os to get CPU cores
    cpu_cores = os.cpu_count()
    print(f"CPU Cores: {cpu_cores}")

    # Check for GPU
    is_gpu_available = torch.cuda.is_available()
    print(f"GPU Available: {is_gpu_available}")

    # Print RAM and storage specs
    # RAM Information
    ram_info = psutil.virtual_memory()
    total_ram_gb = ram_info.total / (1024**3)
    print(f"Total RAM: {total_ram_gb:.2f} GB")

    # Storage Information
    disk_info = psutil.disk_usage('/')
    total_disk_gb = disk_info.total / (1024**3)
    free_disk_gb = disk_info.free / (1024**3)
    print(f"Total Disk Space: {total_disk_gb:.2f} GB")
    print(f"Free Disk Space: {free_disk_gb:.2f} GB")

def parse_filename_datetime(filename):
    """
    Extract date/time range from SpaceNet filename format.
    
    Format: SN6_Train_AOI_11_Rotterdam_PS-RGB_20190823162315_20190823162606_tile_7879
    Time format: YYYYMMDDHHmmss_YYYYMMDDHHmmss (start_end)
    
    Args:
        filename (str): Image filename (basename or full path)
        
    Returns:
        tuple: (start_datetime, end_datetime) or (None, None) if not found
    """
    # Extract basename if full path provided
    basename = os.path.basename(filename)
    
    # Pattern: YYYYMMDDHHmmss_YYYYMMDDHHmmss
    pattern = r'(\d{14})_(\d{14})'
    match = re.search(pattern, basename)
    
    if not match:
        return None, None
    
    start_str = match.group(1)
    end_str = match.group(2)
    
    try:
        start_dt = datetime.strptime(start_str, '%Y%m%d%H%M%S')
        end_dt = datetime.strptime(end_str, '%Y%m%d%H%M%S')
        return start_dt, end_dt
    except ValueError:
        return None, None

def detect_black_bars(image_array, threshold=10, min_bar_size=5, bar_threshold=0.95):
    """
    Detect black bars at the top and bottom of an image.
    
    Args:
        image_array: numpy array of shape (H, W) or (H, W, C)
        threshold: pixel value threshold below which is considered "black" (0-255)
        min_bar_size: minimum number of rows/columns to consider as a bar
        bar_threshold: fraction of pixels in a row/column that must be black to consider it a bar
    
    Returns:
        tuple: (top_crop, bottom_crop, left_crop, right_crop) in pixels
    """
    # Handle multi-channel images by converting to grayscale
    if len(image_array.shape) == 3:
        # Use mean across channels or max to detect black
        gray = np.mean(image_array, axis=2)
    else:
        gray = image_array
    
    height, width = gray.shape
    top_crop = 0
    bottom_crop = 0
    left_crop = 0
    right_crop = 0
    
    # Detect top black bar
    for i in range(height):
        row = gray[i, :]
        black_pixels = np.sum(row < threshold)
        if black_pixels / width >= bar_threshold:
            top_crop = i + 1
        else:
            break
    
    # Detect bottom black bar
    for i in range(height - 1, -1, -1):
        row = gray[i, :]
        black_pixels = np.sum(row < threshold)
        if black_pixels / width >= bar_threshold:
            bottom_crop = height - i
        else:
            break
    
    # Detect left black bar
    for j in range(width):
        col = gray[:, j]
        black_pixels = np.sum(col < threshold)
        if black_pixels / height >= bar_threshold:
            left_crop = j + 1
        else:
            break
    
    # Detect right black bar
    for j in range(width - 1, -1, -1):
        col = gray[:, j]
        black_pixels = np.sum(col < threshold)
        if black_pixels / height >= bar_threshold:
            right_crop = width - j
        else:
            break
    
    # Only return crops if they meet minimum size
    if top_crop < min_bar_size:
        top_crop = 0
    if bottom_crop < min_bar_size:
        bottom_crop = 0
    if left_crop < min_bar_size:
        left_crop = 0
    if right_crop < min_bar_size:
        right_crop = 0
    
    return top_crop, bottom_crop, left_crop, right_crop

def crop_black_bars(image_path, output_path=None, threshold=10, min_bar_size=5, bar_threshold=0.95):
    """
    Crop black bars from an image and optionally save the result.
    
    Args:
        image_path: path to input image
        output_path: optional path to save cropped image (if None, returns array only)
        threshold: pixel value threshold below which is considered "black" (0-255)
        min_bar_size: minimum number of rows/columns to consider as a bar
        bar_threshold: fraction of pixels in a row/column that must be black to consider it a bar
    
    Returns:
        tuple: (cropped_image_array, crop_info) where crop_info is (top, bottom, left, right)
    """
    import rasterio
    
    with rasterio.open(image_path) as src:
        image_array = src.read()
        # Convert from (C, H, W) to (H, W, C) for processing
        if len(image_array.shape) == 3:
            image_array = np.transpose(image_array, (1, 2, 0))
        
        height, width = image_array.shape[:2]
        
        # Detect black bars
        top, bottom, left, right = detect_black_bars(
            image_array, threshold, min_bar_size, bar_threshold
        )
        
        # Crop the image
        if len(image_array.shape) == 3:
            cropped = image_array[top:height-bottom, left:width-right, :]
            # Convert back to (C, H, W) for saving
            cropped_for_save = np.transpose(cropped, (2, 0, 1))
        else:
            cropped = image_array[top:height-bottom, left:width-right]
            cropped_for_save = cropped
        
        new_height = height - top - bottom
        new_width = width - left - right
        
        # Save if output path provided
        if output_path:
            profile = src.profile.copy()
            profile.update({
                'height': new_height,
                'width': new_width,
                'transform': rasterio.Affine(
                    src.transform.a, src.transform.b,
                    src.transform.c + left * src.transform.a,
                    src.transform.d, src.transform.e,
                    src.transform.f + top * src.transform.e
                )
            })
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(cropped_for_save)
        
        crop_info = (top, bottom, left, right)
        return cropped, crop_info

