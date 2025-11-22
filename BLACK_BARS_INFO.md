# Black Bars in SpaceNet 6 Dataset

## Why Black Bars Exist

The black bars (or borders) in SpaceNet 6 Rotterdam images are a known issue that occurs due to several factors:

1. **Sensor Artifacts**: The satellite sensors may capture images with padding or no-data regions at the edges
2. **Tiling Process**: When large satellite images are divided into smaller tiles, some tiles may include edge regions with no actual image data
3. **Data Processing**: During the conversion from raw satellite data to GeoTIFF format, padding or alignment processes can introduce black borders
4. **Coordinate System Alignment**: When images are georeferenced and aligned to specific coordinate systems, empty regions may be filled with zero values (black pixels)

## Impact on Training

Black bars can negatively affect model training because:
- They add noise and reduce the effective training area
- The model learns to associate black regions with "background," which may not generalize well
- They waste computational resources on non-informative pixels
- Bounding boxes may be incorrectly normalized if the black bars are included in the image dimensions

## How Others Have Handled This

### 1. **OpenCV-Based Detection and Cropping**
Many practitioners use OpenCV to automatically detect and crop black borders:

```python
import cv2
import numpy as np

def crop_black_borders_opencv(image_path):
    """Detect and crop black borders using OpenCV."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to create binary mask
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours of non-black regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return img[y:y+h, x:x+w]
    return img
```

### 2. **Row/Column Sum Analysis**
A common approach is to analyze pixel sums along rows and columns:

```python
def detect_black_bars_sums(image_array, threshold=10):
    """Detect black bars by analyzing row/column sums."""
    if len(image_array.shape) == 3:
        gray = np.mean(image_array, axis=2)
    else:
        gray = image_array
    
    # Sum along columns (detect vertical bars)
    col_sums = np.sum(gray, axis=0)
    # Sum along rows (detect horizontal bars)
    row_sums = np.sum(gray, axis=1)
    
    # Find where sums are below threshold
    left = np.where(col_sums > threshold * gray.shape[0])[0][0]
    right = np.where(col_sums > threshold * gray.shape[0])[0][-1]
    top = np.where(row_sums > threshold * gray.shape[1])[0][0]
    bottom = np.where(row_sums > threshold * gray.shape[1])[0][-1]
    
    return top, bottom, left, right
```

### 3. **GDAL-Based Processing**
For geospatial data, GDAL tools are often used:

```bash
# Convert and process GeoTIFFs
gdal_translate -scale 0 65535 0 255 -ot Byte input.tif output.tif
```

### 4. **SpaceNet Utilities**
The official SpaceNet Challenge repository provides preprocessing utilities:
- GitHub: https://github.com/SpaceNetChallenge/utilities
- Includes tools for format conversion, tiling, and preprocessing

## Our Implementation

We've implemented a solution that:
1. **Detects black bars** by analyzing pixel intensity along rows/columns
2. **Adjusts bounding box coordinates** to account for the cropped regions
3. **Normalizes coordinates** using the effective (cropped) image dimensions

The implementation is in:
- `src/utils.py`: `detect_black_bars()` and `crop_black_bars()` functions
- `scripts/convert_labels.py`: Updated to use black bar detection during label conversion
- `colab_training.ipynb`: Updated training notebook with black bar handling

### Key Parameters

- **threshold**: Pixel value below which is considered "black" (default: 10)
- **min_bar_size**: Minimum number of rows/columns to consider as a bar (default: 5)
- **bar_threshold**: Fraction of pixels in a row/column that must be black (default: 0.95)

These parameters can be adjusted based on your specific dataset characteristics.

## Best Practices

1. **Visual Inspection**: Always inspect a sample of processed images to ensure the cropping is working correctly
2. **Preserve Georeferencing**: When cropping, update the geotransform matrix to maintain spatial accuracy
3. **Adjust Labels**: Always adjust bounding box coordinates when cropping images
4. **Test on Validation Set**: Verify that cropping improves model performance on a validation set
5. **Document Parameters**: Keep track of the threshold values that work best for your specific images

## References

- SpaceNet Challenge Utilities: https://github.com/SpaceNetChallenge/utilities
- SpaceNet 6 Dataset Documentation: https://www.spacenet.ai/sn6-challenge/
- GDAL Documentation: https://gdal.org/
