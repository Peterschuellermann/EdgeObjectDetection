"""
Spatial-temporal matching of detected ships with AIS positions.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from geopy.distance import geodesic
from .provider import AISProvider, AISPosition
from ..config import AIS_MATCH_DISTANCE_M, AIS_TIME_BUFFER_SEC


class AISMatch:
    """Represents a match between a detection and an AIS position."""
    def __init__(self, detection_idx: int, ais_position: AISPosition, distance_m: float):
        self.detection_idx = detection_idx
        self.ais_position = ais_position
        self.distance_m = distance_m


class AISMatcher:
    """
    Matches detected ship positions with AIS data using spatial-temporal criteria.
    """
    
    def __init__(self, provider: AISProvider):
        self.provider = provider
        self.match_distance_m = AIS_MATCH_DISTANCE_M
        self.time_buffer_sec = AIS_TIME_BUFFER_SEC
    
    def match_detections(self, detections: List[Dict], filename: str,
                        start_time: datetime, end_time: datetime) -> List[Optional[AISMatch]]:
        """
        Match detections with AIS positions.
        
        Args:
            detections: List of detection dicts with 'latitude' and 'longitude' keys
            filename: Image filename (for context)
            start_time: Image capture start time
            end_time: Image capture end time
            
        Returns:
            List of AISMatch objects (or None for unmatched detections)
        """
        if not detections:
            return []
        
        # Expand time window with buffer
        query_start = start_time - timedelta(seconds=self.time_buffer_sec)
        query_end = end_time + timedelta(seconds=self.time_buffer_sec)
        
        # Get bounding box from detections
        lats = [d['latitude'] for d in detections]
        lons = [d['longitude'] for d in detections]
        
        min_lat = min(lats) - 0.01  # ~1km buffer
        max_lat = max(lats) + 0.01
        min_lon = min(lons) - 0.01
        max_lon = max(lons) + 0.01
        
        
        ais_positions = self.provider.query_positions(
            start_time=query_start,
            end_time=query_end,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon
        )
        
        # Match each detection with closest AIS position
        matches = []
        for i, detection in enumerate(detections):
            det_lat = detection['latitude']
            det_lon = detection['longitude']
            
            best_match = None
            best_distance = float('inf')
            
            for ais_pos in ais_positions:
                # Calculate distance
                distance = geodesic(
                    (det_lat, det_lon),
                    (ais_pos.lat, ais_pos.lon)
                ).meters
                
                # Check if within threshold and time window
                if distance <= self.match_distance_m:
                    # Check if AIS position is within image capture time window
                    if start_time <= ais_pos.timestamp <= end_time:
                        if distance < best_distance:
                            best_distance = distance
                            best_match = AISMatch(i, ais_pos, distance)
            
            matches.append(best_match)
        
        return matches
    
    def match_detections_batch(self, detection_groups: List[Tuple[List[Dict], str, datetime, datetime]]) -> List[List[Optional[AISMatch]]]:
        """
        Match multiple groups of detections (e.g., from multiple images).
        
        Args:
            detection_groups: List of (detections, filename, start_time, end_time) tuples
            
        Returns:
            List of match lists, one per detection group
        """
        all_matches = []
        for detections, filename, start_time, end_time in detection_groups:
            matches = self.match_detections(detections, filename, start_time, end_time)
            all_matches.append(matches)
        return all_matches
