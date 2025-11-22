import glob
import os
import uuid
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils import setup_logging, inspect_environment
from src.data_loader import list_s3_files, download_files_parallel
from src.model_handler import load_model
from src.inference import run_inference
from src.visualization import plot_results, create_map, create_map_by_day
from src.config import LOCAL_DIR
from src.database import init_db, insert_detection, get_detections_by_run_id
from src.analytics.recorder import DetectionRecorder
from src.ais_workflow import match_detections_with_ais
from geopy.distance import geodesic

def chunked(seq, n):
    """Yield successive chunks of size n from seq."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def process_batch(file_batch):
    """
    Runs run_inference_crop on a subset of files in a separate process.
    Each process loads its own detection model once.
    """
    # Local import to avoid issues when this function is pickled
    from src.model_handler import load_model
    from src.inference_crop import run_inference_crop
    
    detection_model = load_model()
    # We don't pass analytics_callbacks here because we handle DB logging in main process
    geo_detections, inference_cache, timings, sahi_timings = run_inference_crop(file_batch, detection_model)
    return geo_detections, inference_cache, timings, sahi_timings

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

    # 3. Run Inference (Parallel)
    folder = LOCAL_DIR
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))#[:100]
    
    print(f"Running SAHI/PyTorch pipeline in parallel on {len(files)} files...")
    print("Starting inference...")
    
    t0 = time.perf_counter()
    
    num_workers = 16 # Conservative default, was 16 in optimization but depends on system
    # Adjust based on CPU count
    try:
        cpu_count = os.cpu_count()
        if cpu_count:
            num_workers = min(16, cpu_count) # Cap at 16 or CPU count
    except:
        pass
        
    if len(files) < num_workers:
        num_workers = len(files)

    if num_workers < 1:
        num_workers = 1

    chunk_size = max(1, len(files) // num_workers)
    
    all_geo_detections = []
    all_inference_cache = {}
    aggregated_timings = {}
    aggregated_sahi_timings = {}

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = []
        for batch in chunked(files, chunk_size):
            futures.append(ex.submit(process_batch, batch))

        for fut in as_completed(futures):
            try:
                geo_dets, cache, timings, sahi_timings = fut.result()
                all_geo_detections.extend(geo_dets)
                all_inference_cache.update(cache)
                
                # Aggregate timings
                for k, v in timings.items():
                    aggregated_timings[k] = aggregated_timings.get(k, 0.0) + v
                for k, v in sahi_timings.items():
                    aggregated_sahi_timings[k] = aggregated_sahi_timings.get(k, 0.0) + v
                    
            except Exception as e:
                print(f"Error in parallel execution: {e}")

    t1 = time.perf_counter()
    print(f"Finished in {t1 - t0:.2f} seconds\n")
    # print(f"\n✅ Total parallel inference wall time: {t1 - t0:.2f} seconds\n")
    
    # Print aggregated timings
    # print("===== Aggregated Inference Timing Summary (pipeline) =====")
    # for k, v in aggregated_timings.items():
    #     print(f"{k:20s}: {v:.4f} seconds")
    # print(f"{'TOTAL (sum of stages)':20s}: {sum(aggregated_timings.values()):.4f} seconds\n")

    # print("===== Aggregated SAHI Timing Summary (inside get_sliced_prediction) =====")
    # for k, v in aggregated_sahi_timings.items():
    #     print(f"{k:20s}: {v:.4f} seconds")
    # print(f"{'TOTAL (SAHI slice+pred+post)':20s}: {sum(aggregated_sahi_timings.values()):.4f} seconds")
    
    geo_detections = all_geo_detections
    inference_cache = all_inference_cache

    # 4. Record to Database
    print("Recording detections to database...")
    for det in geo_detections:
         insert_detection(
                run_id=run_id,
                file_name=det['file'],
                label=det['label'],
                score=det['score'],
                geometry_wkt=det['geometry'].wkt,
                latitude=det['latitude'],
                longitude=det['longitude']
            )

    # 5. AIS Matching
    ais_stats = match_detections_with_ais(run_id, geo_detections, files)
    
    # Update geo_detections with AIS match info for visualization
    # Get updated records from database
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
    
    print("\nGenerating day-specific maps...")
    day_to_filepath = create_map_by_day(files, geo_detections)
    
    print("\nGenerating combined map...")
    create_map(files, geo_detections, output_file="detection_map.html")

if __name__ == "__main__":
    main()
