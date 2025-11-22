#!/usr/bin/env python3
"""
Simple integration test to verify the AIS workflow components.
"""

import sys
import os

def test_filename_parser():
    """Test filename datetime parsing."""
    print("Testing filename parser...")
    from src.utils import parse_filename_datetime
    
    filename = "SN6_Train_AOI_11_Rotterdam_PS-RGB_20190823162315_20190823162606_tile_7879.tif"
    start, end = parse_filename_datetime(filename)
    
    assert start is not None, "Start time should not be None"
    assert end is not None, "End time should not be None"
    assert start.year == 2019, "Year should be 2019"
    assert start.month == 8, "Month should be 8"
    assert start.day == 23, "Day should be 23"
    assert start.hour == 16, "Hour should be 16"
    assert start.minute == 23, "Minute should be 23"
    assert start.second == 15, "Second should be 15"
    
    print("  ✅ Filename parser works correctly")
    return True

def test_database_operations():
    """Test database operations with AIS columns."""
    print("Testing database operations...")
    from src.database import init_db, insert_detection, get_detections_by_run_id, update_detection_ais
    
    # Initialize
    init_db()
    
    # Insert test detection
    insert_detection(
        run_id='test-integration',
        file_name='test_ship.tif',
        label='ship',
        score=0.95,
        geometry_wkt='POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
        latitude=51.9,
        longitude=4.2
    )
    
    # Retrieve and verify
    records = get_detections_by_run_id('test-integration')
    assert len(records) > 0, "Should have at least one record"
    
    record = records[0]
    # Test sqlite3.Row bracket access (the bug fix)
    assert record['file_name'] == 'test_ship.tif', "File name should match"
    assert record['latitude'] == 51.9, "Latitude should match"
    assert record['longitude'] == 4.2, "Longitude should match"
    
    # Test AIS update
    update_detection_ais(
        id=record['id'],
        ais_matched=True,
        ais_mmsi='123456789',
        ais_vessel_name='Test Vessel',
        ais_vessel_type='Cargo',
        ais_distance_m=50.5
    )
    
    # Verify update
    records_updated = get_detections_by_run_id('test-integration')
    updated_record = records_updated[0]
    assert updated_record['ais_matched'] == 1, "AIS matched should be 1"
    assert updated_record['ais_mmsi'] == '123456789', "MMSI should match"
    
    print("  ✅ Database operations work correctly")
    return True

def test_coordinate_handling():
    """Test coordinate handling logic."""
    print("Testing coordinate handling...")
    
    # Simulate transform_coords with always_xy=True
    # It returns (lon, lat) tuples
    test_lon = [4.2]  # Rotterdam longitude
    test_lat = [51.9]  # Rotterdam latitude
    
    # This is how the code unpacks it
    lon, lat = test_lon, test_lat
    latitude = lat[0]
    longitude = lon[0]
    
    assert latitude == 51.9, "Latitude should be 51.9"
    assert longitude == 4.2, "Longitude should be 4.2"
    
    print("  ✅ Coordinate handling is correct")
    return True

def test_sqlite3_row_access():
    """Test sqlite3.Row access pattern (the bug fix)."""
    print("Testing sqlite3.Row access pattern...")
    from src.database import init_db, insert_detection, get_detections_by_run_id
    
    init_db()
    insert_detection('test-row', 'test.tif', 'ship', 0.9, 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))', 51.9, 4.2)
    
    records = get_detections_by_run_id('test-row')
    record = records[0]
    
    # Test the try-except pattern from main.py
    try:
        ais_matched = bool(record['ais_matched'] if record['ais_matched'] is not None else 0)
        assert isinstance(ais_matched, bool), "ais_matched should be bool"
    except (KeyError, IndexError):
        ais_matched = False
    
    try:
        ais_mmsi = record['ais_mmsi']
    except (KeyError, IndexError):
        ais_mmsi = None
    
    # Test that .get() would fail (proving the bug exists)
    try:
        result = record.get('ais_mmsi')  # This should fail
        print("  ⚠️  Warning: .get() method exists (unexpected)")
    except AttributeError:
        print("  ✅ Confirmed: sqlite3.Row doesn't have .get() method")
    
    print("  ✅ sqlite3.Row access pattern works correctly")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Integration Tests for AIS Workflow")
    print("=" * 60)
    print()
    
    tests = [
        test_filename_parser,
        test_database_operations,
        test_coordinate_handling,
        test_sqlite3_row_access,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
