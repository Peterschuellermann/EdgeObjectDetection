"""
AIS matching workflow integration.
"""

import os
from datetime import datetime
from typing import List, Dict
from .utils import parse_filename_datetime
from .ais import AISMatcher, DatalasticProvider, LocalCSVProvider
from .config import AIS_PROVIDER, AIS_API_KEY, AIS_LOCAL_CSV_PATH
from .database import get_detections_by_run_id, update_detection_ais


def create_ais_provider():
    """Create AIS provider based on configuration."""
    if AIS_PROVIDER == "datalastic":
        return DatalasticProvider(api_key=AIS_API_KEY)
    elif AIS_PROVIDER == "local_csv":
        if not AIS_LOCAL_CSV_PATH:
            raise ValueError("AIS_LOCAL_CSV_PATH must be set in config.py for local_csv provider")
        return LocalCSVProvider(csv_path=AIS_LOCAL_CSV_PATH)
    else:
        raise ValueError(f"Unknown AIS provider: {AIS_PROVIDER}")


def match_detections_with_ais(run_id: str, geo_detections: List[Dict], files: List[str]) -> Dict:
    """
    Match detections with AIS data and update database.
    
    Args:
        run_id: Run ID for the detection batch
        geo_detections: List of detection dicts with 'file', 'latitude', 'longitude'
        files: List of image file paths
        
    Returns:
        Dictionary with matching statistics
    """
    print("\n" + "="*60)
    print("Starting AIS Matching Workflow")
    print("="*60)
    
    # Group detections by file
    detections_by_file = {}
    for det in geo_detections:
        filename = det['file']
        if filename not in detections_by_file:
            detections_by_file[filename] = []
        detections_by_file[filename].append(det)
    
    # Create AIS provider and matcher
    try:
        provider = create_ais_provider()
        matcher = AISMatcher(provider)
    except Exception as e:
        print(f"Error creating AIS provider: {e}")
        print("Skipping AIS matching. Set AIS_API_KEY or configure provider.")
        return {"matched": 0, "unknown": len(geo_detections), "error": str(e)}
    
    # Match detections for each file
    total_matched = 0
    total_unknown = 0
    
    # Get database records for this run
    db_records = get_detections_by_run_id(run_id)
    # Create a mapping from (file, label, score) to database ID
    # Use label and score for matching since lat/lon might have precision issues
    db_map = {}
    for record in db_records:
        key = (record['file_name'], record['label'], record['score'])
        db_map[key] = record['id']
    
    for file_path in files:
        filename = os.path.basename(file_path)
        
        if filename not in detections_by_file:
            continue
        
        detections = detections_by_file[filename]
        
        # Filter to only process detections with label 'ship'
        ship_detections = [det for det in detections if str(det.get('label', '')).lower() == 'ship']
        
        if not ship_detections:
            print(f"  No ship detections in {filename}, skipping AIS matching")
            continue
        
        # Use only ship detections for AIS matching
        detections = ship_detections
        
        # Parse timestamp from filename
        start_time, end_time = parse_filename_datetime(filename)
        if start_time is None or end_time is None:
            print(f"Warning: Could not parse timestamp from {filename}, skipping AIS matching")
            continue
        
        print(f"\nProcessing {filename}")
        print(f"  Time range: {start_time} to {end_time}")
        print(f"  Ship detections: {len(detections)}")
        
        # Match detections
        matches = matcher.match_detections(detections, filename, start_time, end_time)
        
        # Update database records
        for i, match in enumerate(matches):
            detection = detections[i]
            
            # Find corresponding database record by file, label, and score
            db_key = (filename, detection['label'], detection['score'])
            if db_key not in db_map:
                # Try to find by file and approximate lat/lon (within 10m tolerance)
                db_id = None
                for record in db_records:
                    if record['file_name'] == filename:
                        try:
                            rec_lat = record['latitude']
                            rec_lon = record['longitude']
                        except (KeyError, IndexError):
                            rec_lat = None
                            rec_lon = None
                        if rec_lat is not None and rec_lon is not None:
                            from geopy.distance import geodesic
                            dist = geodesic(
                                (detection['latitude'], detection['longitude']),
                                (rec_lat, rec_lon)
                            ).meters
                            if dist < 10:  # Within 10 meters
                                db_id = record['id']
                                break
                if db_id is None:
                    continue
            else:
                db_id = db_map[db_key]
            
            if match:
                # Matched with AIS
                update_detection_ais(
                    id=db_id,
                    ais_matched=True,
                    ais_mmsi=match.ais_position.mmsi,
                    ais_vessel_name=match.ais_position.vessel_name,
                    ais_vessel_type=match.ais_position.vessel_type,
                    ais_distance_m=match.distance_m
                )
                total_matched += 1
            else:
                # No match - mark as unknown
                update_detection_ais(
                    id=db_id,
                    ais_matched=False
                )
                total_unknown += 1
    
    print("\n" + "="*60)
    print("AIS Matching Complete (Ships Only)")
    print("="*60)
    print(f"Identified (matched): {total_matched}")
    print(f"Unknown (no match): {total_unknown}")
    print(f"Total ships processed: {total_matched + total_unknown}")
    
    return {
        "matched": total_matched,
        "unknown": total_unknown,
        "total": total_matched + total_unknown
    }
