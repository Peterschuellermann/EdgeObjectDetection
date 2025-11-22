"""
AIS data provider interfaces for querying historical vessel positions.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Optional
import requests
from ..config import AIS_API_KEY


class AISPosition:
    """Represents a single AIS position report."""
    def __init__(self, mmsi: str, lat: float, lon: float, timestamp: datetime,
                 vessel_name: Optional[str] = None, vessel_type: Optional[str] = None,
                 speed: Optional[float] = None, heading: Optional[float] = None):
        self.mmsi = mmsi
        self.lat = lat
        self.lon = lon
        self.timestamp = timestamp
        self.vessel_name = vessel_name
        self.vessel_type = vessel_type
        self.speed = speed
        self.heading = heading


class AISProvider(ABC):
    """Abstract base class for AIS data providers."""
    
    @abstractmethod
    def query_positions(self, start_time: datetime, end_time: datetime,
                       min_lat: float, max_lat: float,
                       min_lon: float, max_lon: float) -> List[AISPosition]:
        """
        Query AIS positions within a geographic bounding box and time window.
        
        Args:
            start_time: Start of time window
            end_time: End of time window
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
            
        Returns:
            List of AISPosition objects
        """
        pass


class DatalasticProvider(AISProvider):
    """
    Datalastic API provider for historical AIS data.
    
    API Documentation: https://datalastic.com/api-doc
    """
    
    BASE_URL = "https://api.datalastic.com/api/v0"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or AIS_API_KEY
        if not self.api_key:
            raise ValueError("Datalastic API key is required. Set AIS_API_KEY in config.py")
    
    def query_positions(self, start_time: datetime, end_time: datetime,
                       min_lat: float, max_lat: float,
                       min_lon: float, max_lon: float) -> List[AISPosition]:
        """
        Query Datalastic API for historical vessel positions.
        
        Note: Datalastic's API may require querying by MMSI or using their
        vessel search endpoint. This is a simplified implementation that
        queries for vessels in the area. For production, you may need to
        adjust based on actual API capabilities.
        """
        positions = []
        
        # Datalastic API endpoint for vessel search
        # Note: This is a simplified implementation. Actual API may differ.
        url = f"{self.BASE_URL}/vessel_search"
        
        # Calculate center point and approximate radius
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        
        params = {
            "api-key": self.api_key,
            "lat": center_lat,
            "lon": center_lon,
            "radius": 50,  # km radius around center
            "from": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "to": end_time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Parse response based on Datalastic API format
            # Note: Actual response structure may vary - adjust as needed
            if "data" in data and isinstance(data["data"], list):
                for vessel_data in data["data"]:
                    # Extract position data
                    # Adjust field names based on actual API response
                    mmsi = str(vessel_data.get("mmsi", ""))
                    lat = float(vessel_data.get("lat", 0))
                    lon = float(vessel_data.get("lon", 0))
                    
                    # Parse timestamp
                    timestamp_str = vessel_data.get("timestamp", "")
                    if timestamp_str:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        except:
                            timestamp = start_time  # Fallback
                    else:
                        timestamp = start_time
                    
                    # Filter by bounding box
                    if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                        position = AISPosition(
                            mmsi=mmsi,
                            lat=lat,
                            lon=lon,
                            timestamp=timestamp,
                            vessel_name=vessel_data.get("name"),
                            vessel_type=vessel_data.get("type"),
                            speed=vessel_data.get("speed"),
                            heading=vessel_data.get("heading")
                        )
                        positions.append(position)
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                 pass
            else:
                 pass
        except requests.exceptions.RequestException as e:
            pass
            # Return empty list on error
        except (KeyError, ValueError, TypeError) as e:
            pass
        
        return positions


class LocalCSVProvider(AISProvider):
    """
    Local CSV file provider for pre-downloaded AIS data.
    
    CSV format expected:
    timestamp,mmsi,lat,lon,vessel_name,vessel_type,speed,heading
    """
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
    
    def query_positions(self, start_time: datetime, end_time: datetime,
                       min_lat: float, max_lat: float,
                       min_lon: float, max_lon: float) -> List[AISPosition]:
        """Query positions from local CSV file."""
        import pandas as pd
        
        try:
            df = pd.read_csv(self.csv_path)
            
            # Parse timestamp column
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by time window
            mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
            df = df[mask]
            
            # Filter by bounding box
            mask = (df['lat'] >= min_lat) & (df['lat'] <= max_lat) & \
                   (df['lon'] >= min_lon) & (df['lon'] <= max_lon)
            df = df[mask]
            
            positions = []
            for _, row in df.iterrows():
                position = AISPosition(
                    mmsi=str(row.get('mmsi', '')),
                    lat=float(row.get('lat', 0)),
                    lon=float(row.get('lon', 0)),
                    timestamp=row['timestamp'].to_pydatetime(),
                    vessel_name=row.get('vessel_name'),
                    vessel_type=row.get('vessel_type'),
                    speed=row.get('speed'),
                    heading=row.get('heading')
                )
                positions.append(position)
            
            return positions
        
        except Exception as e:
            return []
