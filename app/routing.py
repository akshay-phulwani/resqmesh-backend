import math
import httpx
from typing import List, Dict, Any, Tuple

# Haversine formula to compute great-circle distance
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0 # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Distance from a point to a line segment (in degrees, approximate)
def point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    
    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

# Global list of blockages in memory for simplicity (synced with Redis if needed, but in-memory works great for API runtime)
active_blockages: List[Dict[str, Any]] = []

def add_blockage(blockage_id: str, lat: float, lon: float, description: str, radius_m: float = 150.0):
    # Remove if exists
    remove_blockage(blockage_id)
    active_blockages.append({
        "id": blockage_id,
        "latitude": lat,
        "longitude": lon,
        "description": description,
        "radius_meters": radius_m
    })

def remove_blockage(blockage_id: str):
    global active_blockages
    active_blockages = [b for b in active_blockages if b["id"] != blockage_id]

def get_active_blockages() -> List[Dict[str, Any]]:
    return active_blockages

def check_route_blocked(route_coords: List[List[float]]) -> Tuple[bool, float, str]:
    """
    Checks if a route intersects any active blockages.
    Returns (is_blocked, penalty_seconds, description)
    """
    for blockage in active_blockages:
        b_lat, b_lon = blockage["latitude"], blockage["longitude"]
        # Convert meters to approximate degree threshold (111km per degree)
        threshold_deg = blockage["radius_meters"] / 111000.0
        
        # Check if any segment of the route gets too close to the blockage
        for i in range(len(route_coords) - 1):
            p1, p2 = route_coords[i], route_coords[i+1]
            dist = point_to_segment_distance(b_lat, b_lon, p1[0], p1[1], p2[0], p2[1])
            if dist < threshold_deg:
                # Route is blocked! Return heavy penalty
                return True, 1800.0, f"Route blocked by: {blockage['description']}"
                
    return False, 0.0, ""

def generate_mock_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> Dict[str, Any]:
    """
    Generates a realistic zig-zag street-like grid route and calculates distance/duration.
    """
    distance_km = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    # Add a factor of 1.3 to simulate real street winding
    distance_km_route = distance_km * 1.25
    
    # 50 km/h average emergency speed
    duration_sec = (distance_km_route / 45.0) * 3600.0
    
    # Generate zig-zag grid coordinate path: Start -> Midpoint1 -> Midpoint2 -> End
    # This looks like streets on a map rather than a straight line
    mid_lat1 = start_lat
    mid_lon1 = end_lon
    
    # Let's add some jitter to make it look even more street-aligned
    route_geom = [
        [start_lat, start_lon],
        [mid_lat1, (start_lon + end_lon) / 2.0],
        [mid_lat1, end_lon],
        [end_lat, end_lon]
    ]
    
    return {
        "distance_m": distance_km_route * 1000.0,
        "duration_sec": duration_sec,
        "geometry": route_geom
    }

async def calculate_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> Dict[str, Any]:
    """
    Simulated routing engine. Bypasses external OSRM calls to prevent local latency/timeout issues.
    Applies active blockage checks over coordinates.
    """
    mock_data = generate_mock_route(start_lat, start_lon, end_lat, end_lon)
    geometry = mock_data["geometry"]
    is_blocked, penalty, block_desc = check_route_blocked(geometry)
    
    return {
        "distance_m": mock_data["distance_m"],
        "duration_sec": mock_data["duration_sec"] + penalty,
        "geometry": geometry,
        "is_blocked": is_blocked,
        "block_message": block_desc
    }
