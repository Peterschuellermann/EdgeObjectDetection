import os
from shapely.geometry import box
from src.database import insert_detection

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
        
        for det in result.object_prediction_list:
            # Transform bbox to map coordinates
            x1, y1 = transform * (det.bbox.minx, det.bbox.miny)
            x2, y2 = transform * (det.bbox.maxx, det.bbox.maxy)
            
            # Create geometry
            geometry = box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            
            # Extract other fields
            label = det.category.name
            score = round(det.score.value, 2)
            
            # Save to DB
            insert_detection(
                run_id=self.run_id,
                file_name=file_name,
                label=label,
                score=score,
                geometry_wkt=geometry.wkt
            )
