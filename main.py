import glob
import os
import uuid
from src.utils import setup_logging, inspect_environment
from src.data_loader import list_s3_files, download_files_parallel
from src.model_handler import load_model
from src.inference import run_inference
from src.visualization import plot_results, create_map
from src.config import LOCAL_DIR
from src.database import init_db
from src.analytics.recorder import DetectionRecorder
from src.ais_workflow import match_detections_with_ais

def main():
    # 1. Setup
    setup_logging()
    inspect_environment()
    
    # Initialize Database
    init_db()
    run_id = str(uuid.uuid4())
    print(f"Starting Run ID: {run_id}")

    # 2. Data Download
    tasks = list_s3_files()
    download_files_parallel(tasks)

    # 3. Load Model
    model = load_model()

    # 4. Run Inference
    # Select a subset of files for demonstration as in the notebook
    folder = LOCAL_DIR
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))[:100]
    
    # Create recorder callback
    recorder = DetectionRecorder(run_id)
    
    geo_detections, inference_cache = run_inference(files, model, analytics_callbacks=[recorder])

    # 5. AIS Matching
    print("\n" + "="*60)
    print("AIS Ship Identification")
    print("="*60)
    ais_stats = match_detections_with_ais(run_id, geo_detections, files)
    
    # Update geo_detections with AIS match info for visualization
    # Get updated records from database
    from src.database import get_detections_by_run_id
    from geopy.distance import geodesic
    db_records = get_detections_by_run_id(run_id)
    # Create map by (file, label, score) for exact matching
    db_map = {}
    for record in db_records:
        key = (record['file_name'], record['label'], record['score'])
        db_map[key] = record
    
    # Add AIS info to geo_detections
    for det in geo_detections:
        key = (det['file'], det['label'], det['score'])
        if key in db_map:
            record = db_map[key]
            # sqlite3.Row doesn't have .get(), use bracket notation with try-except
            try:
                det['ais_matched'] = bool(record['ais_matched'] if record['ais_matched'] is not None else 0)
            except (KeyError, IndexError):
                det['ais_matched'] = False
            try:
                det['ais_mmsi'] = record['ais_mmsi']
            except (KeyError, IndexError):
                det['ais_mmsi'] = None
            try:
                det['ais_vessel_name'] = record['ais_vessel_name']
            except (KeyError, IndexError):
                det['ais_vessel_name'] = None
            try:
                det['ais_vessel_type'] = record['ais_vessel_type']
            except (KeyError, IndexError):
                det['ais_vessel_type'] = None
            try:
                det['ais_distance_m'] = record['ais_distance_m']
            except (KeyError, IndexError):
                det['ais_distance_m'] = None
        else:
            # Try approximate match by location
            best_match = None
            best_dist = float('inf')
            for record in db_records:
                if record['file_name'] == det['file']:
                    try:
                        rec_lat = record['latitude']
                        rec_lon = record['longitude']
                    except (KeyError, IndexError):
                        rec_lat = None
                        rec_lon = None
                    if rec_lat is not None and rec_lon is not None:
                        dist = geodesic(
                            (det['latitude'], det['longitude']),
                            (rec_lat, rec_lon)
                        ).meters
                        if dist < 10 and dist < best_dist:
                            best_dist = dist
                            best_match = record
            
            if best_match:
                try:
                    det['ais_matched'] = bool(best_match['ais_matched'] if best_match['ais_matched'] is not None else 0)
                except (KeyError, IndexError):
                    det['ais_matched'] = False
                try:
                    det['ais_mmsi'] = best_match['ais_mmsi']
                except (KeyError, IndexError):
                    det['ais_mmsi'] = None
                try:
                    det['ais_vessel_name'] = best_match['ais_vessel_name']
                except (KeyError, IndexError):
                    det['ais_vessel_name'] = None
                try:
                    det['ais_vessel_type'] = best_match['ais_vessel_type']
                except (KeyError, IndexError):
                    det['ais_vessel_type'] = None
                try:
                    det['ais_distance_m'] = best_match['ais_distance_m']
                except (KeyError, IndexError):
                    det['ais_distance_m'] = None
            else:
                det['ais_matched'] = False

    # 6. Visualization
    print("Generating plots...")
    plot_results(files, inference_cache)
    
    print("Generating map...")
    create_map(files, geo_detections, output_file="detection_map.html")

if __name__ == "__main__":
    main()
