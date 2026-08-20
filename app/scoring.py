from typing import Dict, Any, List
from .models import Resource, Hospital, Incident
from .routing import calculate_route

# Map incident types to required hospital specialties
SPECIALTY_MAP = {
    "Cardiac Arrest": "Cardiac",
    "Heart Attack": "Cardiac",
    "Chest Pain": "Cardiac",
    "Stroke": "Trauma",
    "Structure Fire": "Burn",
    "Burn": "Burn",
    "Gas Leak": "Trauma",
    "Car Collision": "Trauma",
    "Multi-vehicle Collision": "Trauma",
    "Gunshot Wound": "Trauma",
    "Assault": "Trauma",
    "Pediatric Emergency": "Pediatric",
    "Choking (Child)": "Pediatric",
    "Seizure": "Trauma",
}

async def calculate_trust_score(
    incident: Incident,
    resource: Resource,
    hospital: Hospital
) -> Dict[str, Any]:
    """
    Computes a deterministic TrustScore (0-100) for a resource-hospital recommendation.
    """
    # 1. Calculate Routes
    # Route 1: Resource -> Incident
    route1 = await calculate_route(resource.latitude, resource.longitude, incident.latitude, incident.longitude)
    # Route 2: Incident -> Hospital
    route2 = await calculate_route(incident.latitude, incident.longitude, hospital.latitude, hospital.longitude)
    
    # 2. Check Blockages
    is_route_blocked = route1.get("is_blocked", False) or route2.get("is_blocked", False)
    
    if is_route_blocked:
        # Route is blocked, completely unusable recommendation
        return {
            "trust_score": 0.0,
            "score_breakdown": {
                "distance_score": 0.0,
                "time_score": 0.0,
                "capacity_score": 0.0,
                "specialty_score": 0.0,
                "route_condition_score": 0.0,
                "total_score": 0.0
            },
            "route_geometry": route1["geometry"] + route2["geometry"],
            "duration_sec": route1["duration_sec"] + route2["duration_sec"],
            "distance_m": route1["distance_m"] + route2["distance_m"],
            "ambulance_coords": [resource.latitude, resource.longitude],
            "hospital_coords": [hospital.latitude, hospital.longitude],
            "explanation": f"Unsuitable option: The route is currently blocked ({route1.get('block_message') or route2.get('block_message')})."
        }

    # 3. Calculate Distance Score (0-100)
    total_dist_m = route1["distance_m"] + route2["distance_m"]
    total_dist_km = total_dist_m / 1000.0
    # Penalty: -5 points per km away from ideal (0 km)
    distance_score = max(0.0, 100.0 - (total_dist_km * 5.0))

    # 4. Calculate Time Score (0-100)
    total_duration_sec = route1["duration_sec"] + route2["duration_sec"]
    total_duration_min = total_duration_sec / 60.0
    # Penalty: -2.5 points per minute of travel time
    time_score = max(0.0, 100.0 - (total_duration_min * 2.5))

    # 5. Calculate Capacity Score (0-100)
    available_capacity = max(0, hospital.emergency_capacity - hospital.current_occupancy)
    # Full capacity = 0 score. Otherwise reward for available capacity, capping at 100 (for 20+ available beds)
    capacity_score = min(100.0, available_capacity * 5.0)

    # 6. Calculate Specialty Match Score (0-100)
    required_specialty = SPECIALTY_MAP.get(incident.incident_type, "Trauma")
    specialty_score = 0.0
    if required_specialty in hospital.specialties:
        specialty_score = 100.0
    elif "Trauma" in hospital.specialties:
        # Trauma is a partial match for general emergencies
        specialty_score = 50.0

    # 7. Route Condition Score (0-100)
    # Check if OSRM added any delay, or check route duration vs base duration
    # Base duration if speed was 50km/h: (distance_km / 50) * 3600.
    base_duration_sec = (total_dist_km / 50.0) * 3600.0
    delay_ratio = total_duration_sec / max(1.0, base_duration_sec)
    
    if delay_ratio <= 1.1:
        route_condition_score = 100.0
    elif delay_ratio <= 1.5:
        route_condition_score = 70.0
    else:
        route_condition_score = 40.0

    # 8. Compute Weighted Total Score
    # Weights: Time (30%), Specialty Match (25%), Distance (20%), Capacity (15%), Route Conditions (10%)
    total_score = (
        0.30 * time_score +
        0.25 * specialty_score +
        0.20 * distance_score +
        0.15 * capacity_score +
        0.10 * route_condition_score
    )
    total_score = round(total_score, 1)

    # Format Explanation
    explanation_parts = [
        f"Medic response time is estimated at {round(route1['duration_sec']/60, 1)} minutes over {round(route1['distance_m']/1000, 1)} km.",
        f"Hospital transit is {round(route2['duration_sec']/60, 1)} minutes."
    ]
    if specialty_score == 100.0:
        explanation_parts.append(f"Hospital matches required specialty ({required_specialty}).")
    elif specialty_score == 50.0:
        explanation_parts.append("Hospital matches trauma fallback.")
    else:
        explanation_parts.append(f"Hospital lacks preferred specialty ({required_specialty}).")
        
    explanation_parts.append(f"Hospital has {available_capacity} available emergency beds.")
    
    explanation = " ".join(explanation_parts)

    return {
        "trust_score": total_score,
        "score_breakdown": {
            "distance_score": round(distance_score, 1),
            "time_score": round(time_score, 1),
            "capacity_score": round(capacity_score, 1),
            "specialty_score": round(specialty_score, 1),
            "route_condition_score": round(route_condition_score, 1),
            "total_score": total_score
        },
        "route_geometry": route1["geometry"] + route2["geometry"],
        "duration_sec": total_duration_sec,
        "distance_m": total_dist_m,
        "ambulance_coords": [resource.latitude, resource.longitude],
        "hospital_coords": [hospital.latitude, hospital.longitude],
        "explanation": explanation
    }
