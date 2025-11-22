import os
import rasterio
from shapely.geometry import box
from rasterio.warp import transform as transform_coords
from src.database import insert_detection
from src.config import DST_CRS

class DetectionRecorder:
    def __init__(self, run_id):
        self.run_id = run_id

    def __call__(self, path, result, img, transform):
        """
        Callback to save detections to the database.
        
        Args:
            path (str): Path to the image file.
            result: SAHI prediction result object.
            img: Image array (unused here but part of signature).
            transform: Affine transform from rasterio.
        """
        file_name = os.path.basename(path)
        
        # Get CRS from the image file
        with rasterio.open(path) as src:
            src_crs = src.crs
        
        for det in result.object_prediction_list:
            # Transform bbox to map coordinates
            x1, y1 = transform * (det.bbox.minx, det.bbox.miny)
            x2, y2 = transform * (det.bbox.maxx, det.bbox.maxy)
            
            # Create geometry
            geometry = box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            
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
            
            # Extract other fields
            label = det.category.name
            score = round(det.score.value, 2)
            
            # Save to DB with lat/lon
            insert_detection(
                run_id=self.run_id,
                file_name=file_name,
                label=label,
                score=score,
                geometry_wkt=geometry.wkt,
                latitude=lat[0],
                longitude=lon[0]
            )
