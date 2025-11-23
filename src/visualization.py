import os
import io
import base64
import hashlib
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import array_bounds
from PIL import Image
import folium
import geopandas as gpd
from sahi.utils.cv import visualize_object_predictions
from .config import DST_CRS, MAPS_OUTPUT_DIR
from .utils import parse_filename_datetime

def plot_results(image_paths, inference_cache, max_plots=10, output_file=None):
    """
    Plots detections for a subset of images and saves to file (non-blocking).
    
    Args:
        image_paths: List of image file paths
        inference_cache: Dictionary mapping image paths to inference results
        max_plots: Maximum number of images to plot
        output_file: Path to save the plot. If None, saves to 'detection_plots.png'
    """
    subset_paths = image_paths[:max_plots]
    
    # Calculate grid dimensions
    n = len(subset_paths)
    cols = 5
    rows = (n + cols - 1) // cols
    
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
    
    # Save plot to file instead of showing (non-blocking)
    if output_file is None:
        output_file = "detection_plots.png"
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close(fig)  # Close figure to free memory

def group_images_by_day(image_paths):
    """
    Groups image paths by day based on the date extracted from filenames.
    
    Args:
        image_paths (list): List of image file paths
        
    Returns:
        dict: Dictionary mapping day strings (YYYY-MM-DD) to lists of image paths
    """
    day_groups = {}
    
    for path in image_paths:
        filename = os.path.basename(path)
        start_dt, end_dt = parse_filename_datetime(filename)
        
        if start_dt is not None:
            day_str = start_dt.strftime('%Y-%m-%d')
            if day_str not in day_groups:
                day_groups[day_str] = []
            day_groups[day_str].append(path)
        else:
            # Handle images without parseable dates
            if 'unknown' not in day_groups:
                day_groups['unknown'] = []
            day_groups['unknown'].append(path)
    
    return day_groups

def get_map_files(output_dir=None):
    """
    Scans the maps directory for day-specific map files.
    
    Args:
        output_dir (str, optional): Directory to scan. Defaults to MAPS_OUTPUT_DIR from config.
        
    Returns:
        dict: Dictionary mapping day strings (YYYY-MM-DD) to file paths
    """
    if output_dir is None:
        output_dir = MAPS_OUTPUT_DIR
    
    if not os.path.exists(output_dir):
        return {}
    
    map_files = {}
    pattern = os.path.join(output_dir, "map_*.html")
    
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        # Extract date from filename: map_YYYY-MM-DD.html
        if filename.startswith("map_") and filename.endswith(".html"):
            day_str = filename[4:-5]  # Remove "map_" prefix and ".html" suffix
            # Validate date format
            if len(day_str) == 10 and day_str.count('-') == 2:
                map_files[day_str] = filepath
    
    return map_files

def save_map_index(day_to_filepath, output_dir=None):
    """
    Saves a JSON index mapping days to map file paths.
    
    Args:
        day_to_filepath (dict): Dictionary mapping day strings to file paths
        output_dir (str, optional): Directory to save index. Defaults to MAPS_OUTPUT_DIR from config.
    """
    if output_dir is None:
        output_dir = MAPS_OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "index.json")
    
    with open(index_path, 'w') as f:
        json.dump(day_to_filepath, f, indent=2)

def create_map(image_paths, geo_detections, output_file="map.html", day_label=None):
    """Creates a Folium map with image overlays and detection boxes."""
    if not image_paths:
        print("No images to map.")
        return

    # Initialize map centered on the first image
    with rasterio.open(image_paths[0]) as src:
        source_crs = src.crs
        transform, width, height = calculate_default_transform(src.crs, DST_CRS, src.width, src.height, *src.bounds)
        left, bottom, right, top = array_bounds(height, width, transform)
        center = [(bottom + top) / 2, (left + right) / 2]

    m = folium.Map(location=center, zoom_start=18, tiles='Esri.WorldImagery')

    # Inject the pixelated style once to avoid repetition
    pixelated_style = """
        <style>
            .leaflet-image-layer {
                /* old android/safari*/
                image-rendering: -webkit-optimize-contrast;
                image-rendering: crisp-edges; /* safari */
                image-rendering: pixelated; /* chrome */
                image-rendering: -moz-crisp-edges; /* firefox */
                image-rendering: -o-crisp-edges; /* opera */
                -ms-interpolation-mode: nearest-neighbor; /* ie */
            }
        </style>
    """
    m.get_root().header.add_child(folium.Element(pixelated_style))

    print(f"Adding {len(image_paths)} images to map...")
    for i, tiff_path in enumerate(image_paths):
        try:
            with rasterio.open(tiff_path) as src:
                nodata_val = src.nodata if src.nodata is not None else 0

                transform, width, height = calculate_default_transform(
                    src.crs, DST_CRS, src.width, src.height, *src.bounds)

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

                layer_name = f"Image {i}" if day_label is None else f"{day_label} - Image {i}"
                folium.raster_layers.ImageOverlay(
                    image=f"data:image/png;base64,{encoded}",
                    bounds=bounds,
                    name=layer_name,
                    pixelated=False
                ).add_to(m)
        except Exception as e:
            print(f"Failed to add image {tiff_path} to map: {e}")

    # Add detections
    if geo_detections:
        gdf = gpd.GeoDataFrame(geo_detections, crs=source_crs)
        gdf = gdf.to_crs(epsg=4326)

        # Color mapping for different object types
        def get_color_for_label(label):
            """Get color for different object types."""
            color_map = {
                'ship': 'blue',
                'vessel': 'blue',
                'boat': 'cyan',
                'building': 'orange',
                'car': 'yellow',
                'truck': 'purple',
                'airplane': 'pink',
                'helicopter': 'magenta',
            }
            # Normalize label to lowercase for matching
            label_lower = str(label).lower()
            # Check for partial matches (e.g., "ship" in "cargo ship")
            for key, color in color_map.items():
                if key in label_lower:
                    return color
            # Default color for unknown types
            return 'red'

        def get_color(feature):
            """Color-code by object type (label) and AIS match status."""
            props = feature['properties']
            label = props.get('label', 'unknown')
            ais_matched = props.get('ais_matched', False)
            # Handle both boolean and integer (0/1) values
            if isinstance(ais_matched, (int, float)):
                ais_matched = bool(ais_matched)
            
            # Check if it's a ship/vessel/boat and AIS matched - make it green
            label_lower = str(label).lower()
            is_ship = any(key in label_lower for key in ['ship', 'vessel', 'boat'])
            
            if is_ship and ais_matched:
                return 'green'
            
            return get_color_for_label(label)
        
        def get_tooltip_fields(feature):
            """Build tooltip text with AIS information if available."""
            props = feature['properties']
            fields = ['label', 'score']
            aliases = ['Object:', 'Confidence:']
            
            if props.get('ais_matched', False):
                fields.extend(['ais_vessel_name', 'ais_mmsi', 'ais_vessel_type', 'ais_distance_m'])
                aliases.extend(['Vessel Name:', 'MMSI:', 'Vessel Type:', 'Distance (m):'])
            else:
                fields.append('ais_matched')
                aliases.append('Status:')
            
            return fields, aliases
        
        # Create tooltip with dynamic fields
        tooltip_fields = ['label', 'score']
        tooltip_aliases = ['Object:', 'Confidence:']
        
        # Check if any detection has AIS info to determine tooltip fields
        # Also check if AIS columns exist in the GeoDataFrame
        has_ais = any(det.get('ais_matched', False) for det in geo_detections)
        gdf_has_ais = any(col.startswith('ais_') for col in gdf.columns)
        
        if has_ais or gdf_has_ais:
            tooltip_fields = ['label', 'score']
            tooltip_aliases = ['Object:', 'Confidence:']
            # Add AIS fields if they exist in the dataframe
            for field in ['ais_matched', 'ais_vessel_name', 'ais_mmsi', 'ais_vessel_type', 'ais_distance_m']:
                if field in gdf.columns:
                    tooltip_fields.append(field)
                    if field == 'ais_matched':
                        tooltip_aliases.append('AIS Matched:')
                    elif field == 'ais_vessel_name':
                        tooltip_aliases.append('Vessel Name:')
                    elif field == 'ais_mmsi':
                        tooltip_aliases.append('MMSI:')
                    elif field == 'ais_vessel_type':
                        tooltip_aliases.append('Vessel Type:')
                    elif field == 'ais_distance_m':
                        tooltip_aliases.append('Distance (m):')
        
        folium.GeoJson(
            gdf,
            name="Detections",
            style_function=lambda x: {
                'color': get_color(x),
                'weight': 3,
                'fillOpacity': 0.3,
                'fillColor': get_color(x)
            },
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True
            ),
            popup=folium.GeoJsonPopup(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True
            )
        ).add_to(m)
        
        # Add legend with object types
        # Get unique labels from detections
        unique_labels = gdf['label'].unique() if 'label' in gdf.columns else []
        legend_items = []
        for label in sorted(unique_labels):
            color = get_color_for_label(label)
            legend_items.append(f'<p><i class="fa fa-circle" style="color:{color}"></i> {label}</p>')
        
        # Check if there are any AIS-matched ships
        has_ais_ships = False
        if 'ais_matched' in gdf.columns:
            ship_labels = ['ship', 'vessel', 'boat']
            for label in unique_labels:
                label_lower = str(label).lower()
                if any(ship in label_lower for ship in ship_labels):
                    # Check for both boolean True and integer 1
                    ais_matched_mask = (gdf['ais_matched'] == True) | (gdf['ais_matched'] == 1)
                    if gdf[(gdf['label'] == label) & ais_matched_mask].shape[0] > 0:
                        has_ais_ships = True
                        break
        
        ais_note = '<p style="margin-top: 10px; font-size: 12px; color: #666;"><i class="fa fa-circle" style="color:green"></i> <strong>Green</strong> = AIS-matched ships</p>' if has_ais_ships else ''
        
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: auto; max-height: 400px; overflow-y: auto;
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>Object Types</h4>
        {''.join(legend_items)}
        {ais_note}
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add JavaScript for filtering based on dashboard selected labels
        filter_js = '''
        <script>
        (function() {
            // Wait for map to be fully loaded
            window.addEventListener('load', function() {
                // Find the detections layer
                var detectionsLayer = null;
                var map = null;
                
                // Find map by looking for Leaflet map instances
                function findMap() {
                    // Try common variable names
                    if (window.map) {
                        map = window.map;
                        return map;
                    }
                    
                    // Find map by looking for divs with leaflet-container class
                    var mapDivs = document.querySelectorAll('.leaflet-container');
                    if (mapDivs.length > 0) {
                        // Try to find map instance from the div
                        for (var i = 0; i < mapDivs.length; i++) {
                            var div = mapDivs[i];
                            // Check if div has a _leaflet_id
                            if (div._leaflet_id !== undefined) {
                                // Try to find the map in L._mapInstances or iterate through window
                                for (var key in window) {
                                    try {
                                        var obj = window[key];
                                        if (obj instanceof L.Map && obj.getContainer() === div) {
                                            map = obj;
                                            return map;
                                        }
                                    } catch(e) {}
                                }
                            }
                        }
                    }
                    
                    // Last resort: find any L.Map instance
                    if (window.L && L.Map) {
                        for (var key in window) {
                            try {
                                if (window[key] instanceof L.Map) {
                                    map = window[key];
                                    return map;
                                }
                            } catch(e) {}
                        }
                    }
                    
                    return null;
                }
                
                // Find detections layer
                function findDetectionsLayer() {
                    map = findMap();
                    if (!map) return null;
                    
                    detectionsLayer = null;
                    
                    // Iterate through all layers
                    map.eachLayer(function(layer) {
                        if (layer instanceof L.GeoJSON) {
                            var name = layer.options && layer.options.name;
                            if (name === 'Detections' || (name && name.includes('Detection'))) {
                                detectionsLayer = layer;
                                return false; // break
                            }
                        }
                    });
                    
                    // Also check _layers directly
                    if (!detectionsLayer && map._layers) {
                        for (var id in map._layers) {
                            var layer = map._layers[id];
                            if (layer instanceof L.GeoJSON) {
                                var name = layer.options && layer.options.name;
                                if (name === 'Detections' || (name && name.includes('Detection'))) {
                                    detectionsLayer = layer;
                                    break;
                                }
                            }
                        }
                    }
                    
                    return detectionsLayer;
                }
                
                // Filter detections based on dashboard selected labels
                function filterDetections() {
                    // Get selected labels from dashboard (injected by dashboard.py)
                    var selectedLabels = window.DASHBOARD_SELECTED_LABELS || [];
                    
                    if (!detectionsLayer) return;
                    
                    detectionsLayer.eachLayer(function(layer) {
                        if (layer.feature && layer.feature.properties) {
                            var label = layer.feature.properties.label || '';
                            // Show if no labels selected (show all) or if label is in selected list
                            if (selectedLabels.length === 0 || selectedLabels.includes(label)) {
                                if (layer._path) {
                                    layer.setStyle({opacity: 1, fillOpacity: 0.3});
                                } else {
                                    layer.setOpacity(1);
                                }
                            } else {
                                if (layer._path) {
                                    layer.setStyle({opacity: 0, fillOpacity: 0});
                                } else {
                                    layer.setOpacity(0);
                                }
                            }
                        }
                    });
                }
                
                // Try multiple times to find the layer and apply filter
                var attempts = 0;
                var findLayerInterval = setInterval(function() {
                    detectionsLayer = findDetectionsLayer();
                    if (detectionsLayer || attempts++ > 10) {
                        clearInterval(findLayerInterval);
                        if (detectionsLayer) {
                            // Apply filter immediately
                            filterDetections();
                            
                            // Also watch for changes to DASHBOARD_SELECTED_LABELS
                            // (in case the dashboard updates it)
                            var lastLabels = JSON.stringify(window.DASHBOARD_SELECTED_LABELS || []);
                            setInterval(function() {
                                var currentLabels = JSON.stringify(window.DASHBOARD_SELECTED_LABELS || []);
                                if (currentLabels !== lastLabels) {
                                    lastLabels = currentLabels;
                                    filterDetections();
                                }
                            }, 500);
                        }
                    }
                }, 500);
            });
        })();
        </script>
        '''
        m.get_root().html.add_child(folium.Element(filter_js))

    folium.LayerControl().add_to(m)
    
    # Save map
    m.save(output_file)
    print(f"Map saved to {output_file}")
    return m

def create_map_by_day(image_paths, geo_detections, output_dir=None):
    """
    Creates separate HTML maps for each day in the dataset.
    
    Args:
        image_paths (list): List of image file paths
        geo_detections (list): List of detection dictionaries with 'file' field
        output_dir (str, optional): Directory to save maps. Defaults to MAPS_OUTPUT_DIR from config.
        
    Returns:
        dict: Dictionary mapping day strings (YYYY-MM-DD) to file paths
    """
    if output_dir is None:
        output_dir = MAPS_OUTPUT_DIR
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group images by day
    day_groups = group_images_by_day(image_paths)
    
    # Filter out 'unknown' day if it exists (or handle it separately)
    day_to_filepath = {}
    
    for day_str, day_image_paths in day_groups.items():
        if day_str == 'unknown':
            # Skip images without parseable dates
            print(f"Skipping {len(day_image_paths)} images with unparseable dates")
            continue
        
        if not day_image_paths:
            continue
        
        # Filter detections for this day
        day_filenames = {os.path.basename(path) for path in day_image_paths}
        day_detections = [
            det for det in geo_detections 
            if det.get('file') in day_filenames
        ]
        
        # Create output filename
        output_file = os.path.join(output_dir, f"map_{day_str}.html")
        
        # Create map for this day
        print(f"\nCreating map for {day_str} ({len(day_image_paths)} images, {len(day_detections)} detections)...")
        create_map(day_image_paths, day_detections, output_file=output_file, day_label=day_str)
        
        day_to_filepath[day_str] = output_file
    
    # Save index file
    if day_to_filepath:
        save_map_index(day_to_filepath, output_dir)
        print(f"\nCreated {len(day_to_filepath)} day-specific maps in {output_dir}")
    
    return day_to_filepath

