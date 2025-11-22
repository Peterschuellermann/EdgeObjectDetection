import os
import io
import base64
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import array_bounds
from PIL import Image
import folium
import geopandas as gpd
from sahi.utils.cv import visualize_object_predictions
from .config import DST_CRS

import os
import matplotlib
matplotlib.use("Agg")  # use non-GUI backend (important for scripts)
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from sahi.utils.cv import visualize_object_predictions

def plot_results(image_paths, inference_cache, max_plots=10, output_dir="plots"):
    """Plots detections for a subset of images and saves the grid to disk."""
    os.makedirs(output_dir, exist_ok=True)

    subset_paths = image_paths[:max_plots]
    
    # Calculate grid dimensions
    n = len(subset_paths)
    cols = 5
    rows = (n + cols - 1) // cols  # ceil(n / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, current_path in enumerate(subset_paths):
        # 1. Load the image for display
        with rasterio.open(current_path) as src:
            img = src.read([1, 2, 3]).transpose(1, 2, 0)

        # 2. Normalize for visuals
        p2, p98 = np.percentile(img, (2, 98))
        div = (p98 - p2) if (p98 - p2) > 0 else 1
        img = np.clip((img - p2) / div * 255.0, 0, 255).astype(np.uint8)
        img_contiguous = np.ascontiguousarray(img)

        # 3. Retrieve the result
        result = inference_cache[current_path]

        # 4. Draw the boxes onto the image using the predictions
        vis_output = visualize_object_predictions(
            image=img_contiguous,
            object_prediction_list=result.object_prediction_list
        )
        vis_img = vis_output["image"]

        # 5. Plot the result
        ax = axes[i]
        ax.imshow(vis_img)
        ax.axis('off')
        ax.set_title(os.path.basename(current_path), fontsize=8)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    # 🔹 Save instead of show
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(output_dir, f"detections_grid_{timestamp}.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved detection grid to: {out_path}")


from datetime import datetime
import os
import io
import base64
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import array_bounds
from PIL import Image
import folium
import geopandas as gpd
from sahi.utils.cv import visualize_object_predictions
from .config import DST_CRS

def create_map(image_paths, geo_detections, output_dir="outputs/maps"):
    """Creates a Folium map with image overlays and detection boxes."""
    if not image_paths:
        print("No images to map.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    html_path = os.path.join(output_dir, f"detection_map_{timestamp}.html")

    # Initialize map centered on the first image
    with rasterio.open(image_paths[0]) as src:
        source_crs = src.crs
        transform, width, height = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds
        )
        left, bottom, right, top = array_bounds(height, width, transform)
        center = [(bottom + top) / 2, (left + right) / 2]

    m = folium.Map(location=center, zoom_start=18, tiles='Esri.WorldImagery')

    print(f"Adding {len(image_paths)} images to map...")
    for i, tiff_path in enumerate(image_paths):
        try:
            with rasterio.open(tiff_path) as src:
                nodata_val = src.nodata if src.nodata is not None else 0

                transform, width, height = calculate_default_transform(
                    src.crs, DST_CRS, src.width, src.height, *src.bounds
                )

                destination = np.zeros((src.count, height, width), dtype=src.dtypes[0])

                reproject(
                    source=rasterio.band(src, range(1, src.count + 1)),
                    destination=destination,
                    dst_transform=transform,
                    dst_crs=DST_CRS,
                    dst_nodata=nodata_val,
                    resampling=Resampling.bilinear
                )

                img_data = np.moveaxis(destination, 0, -1)
                
                # Contrast stretching
                scaled_data = (img_data.astype(np.float32) - 23) * 255.0 / (150 - 23)
                scaled_data = np.clip(scaled_data, 0, 255).astype(np.uint8)

                # Transparency
                mask = (destination[0] == nodata_val)
                alpha = np.where(mask, 0, 255).astype(np.uint8)

                if scaled_data.shape[2] == 1:
                    rgb_data = np.dstack((scaled_data, scaled_data, scaled_data))
                else:
                    rgb_data = scaled_data[..., :3]

                rgba_data = np.dstack((rgb_data, alpha))
                img = Image.fromarray(rgba_data, 'RGBA')

                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                
                b_left, b_bottom, b_right, b_top = array_bounds(height, width, transform)
                bounds = [[b_bottom, b_left], [b_top, b_right]]

                encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')

                folium.raster_layers.ImageOverlay(
                    image=f"data:image/png;base64,{encoded}",
                    bounds=bounds,
                    name=f"Image {i}"
                ).add_to(m)
        except Exception as e:
            print(f"Failed to add image {tiff_path} to map: {e}")

    # Add detections
    if geo_detections:
        gdf = gpd.GeoDataFrame(geo_detections, crs=source_crs)
        gdf = gdf.to_crs(epsg=4326)

        def get_color(label):
            return '#' + hashlib.md5(str(label).encode()).hexdigest()[:6]

        folium.GeoJson(
            gdf,
            name="Detections",
            style_function=lambda x: {
                'color': get_color(x['properties']['label']),
                'weight': 2,
                'fillOpacity': 0.2,
                'fillColor': get_color(x['properties']['label'])
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['label', 'score'],
                aliases=['Object:', 'Confidence:'],
                localize=True
            )
        ).add_to(m)

    folium.LayerControl().add_to(m)
    
    # Save map
    m.save(html_path)
    print(f"Map saved to {html_path}")
    return m

