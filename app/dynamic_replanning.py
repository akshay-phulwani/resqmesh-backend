import json
import asyncio
from typing import List, Dict, Any
from fastapi import WebSocket
from sqlalchemy.orm import Session
from .models import Incident, Resource, Hospital, Recommendation, EmergencyEvent
from .scoring import calculate_trust_score
from .ai_service import explain_recommendation

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        # Maps user/role to connections
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"WebSocket client disconnected. Remaining: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        payload = json.dumps(message)
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception as e:
                print(f"Error sending WebSocket message: {e}")
                # We don't remove during iteration to avoid modifying list, disconnect handles it

# Instantiate a global connection manager
ws_manager = ConnectionManager()

async def recalculate_recommendations(db: Session, incident_id: int) -> List[Recommendation]:
    """
    Recalculates trust scores and ranks resources/hospitals for a given incident.
    Calculates routes and scoring in parallel to optimize response times.
    """
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        return []

    # Get available resources and hospitals
    resources = db.query(Resource).filter(Resource.availability == True).all()
    if not resources:
        resources = db.query(Resource).all()
        
    hospitals = db.query(Hospital).filter(Hospital.availability == True).all()
    if not hospitals:
        hospitals = db.query(Hospital).all()

    # 1. Calculate Trust Scores for all combinations in parallel
    tasks = []
    combinations = []
    for resource in resources:
        for hospital in hospitals:
            combinations.append((resource, hospital))
            tasks.append(calculate_trust_score(incident, resource, hospital))
            
    scores_data = await asyncio.gather(*tasks)
    
    recommendations_list = []
    for (resource, hospital), score_data in zip(combinations, scores_data):
        recommendations_list.append({
            "resource": resource,
            "hospital": hospital,
            "score_data": score_data
        })

    # Sort recommendations by trust score in descending order
    recommendations_list.sort(key=lambda x: x["score_data"]["trust_score"], reverse=True)
    
    # Take the top 3 options
    top_options = recommendations_list[:3]

    # 2. Generate AI explanations in parallel ONLY for the top 3 options
    exp_tasks = [
        explain_recommendation(
            incident_desc=incident.description,
            incident_type=incident.incident_type,
            ambulance_name=opt["resource"].name,
            hospital_name=opt["hospital"].name,
            eta_mins=opt["score_data"]["duration_sec"] / 60.0
        ) for opt in top_options
    ]
    explanations = await asyncio.gather(*exp_tasks)
    
    for opt, explanation in zip(top_options, explanations):
        opt["explanation"] = explanation

    # Delete old recommendations for this incident
    db.query(Recommendation).filter(Recommendation.incident_id == incident_id).delete()
    
    saved_recommendations = []
    for option in top_options:
        res = option["resource"]
        hosp = option["hospital"]
        # Inject duration_sec into score_breakdown
        s_data["score_breakdown"]["duration_sec"] = s_data["duration_sec"]
        
        rec = Recommendation(
            incident_id=incident_id,
            resource_id=res.id,
            hospital_id=hosp.id,
            route_id=f"route_{res.id}_{hosp.id}",
            trust_score=s_data["trust_score"],
            score_breakdown=s_data["score_breakdown"],
            explanation=option["explanation"],
            approval_status="Pending"
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        
        # Attach temporary runtime spatial fields for return schema
        rec.resource_name = res.name
        rec.resource_type = res.type
        rec.hospital_name = hosp.name
        rec.ambulance_coords = s_data["ambulance_coords"]
        rec.hospital_coords = s_data["hospital_coords"]
        rec.route_geometry = s_data["route_geometry"]
        
        saved_recommendations.append(rec)

    # Log dynamic re-planning event
    event = EmergencyEvent(
        incident_id=incident_id,
        event_type="Recommendation_Generated",
        event_metadata={"count": len(saved_recommendations)}
    )
    db.add(event)
    db.commit()

    return saved_recommendations

async def trigger_dynamic_replanning(db: Session, change_description: str):
    """
    Called when a road block is added, or ambulance/hospital becomes unavailable.
    Recalculates recommendations for all active/pending incidents and broadcasts updates.
    """
    # Active incidents are Pending or Dispatched
    active_incidents = db.query(Incident).filter(Incident.status.in_(["Pending", "Dispatched"])).all()
    
    updates = []
    for incident in active_incidents:
        recs = await recalculate_recommendations(db, incident.id)
        
        # Format for WebSocket payload
        rec_data = []
        for r in recs:
            rec_data.append({
                "id": r.id,
                "incident_id": r.incident_id,
                "resource_id": r.resource_id,
                "resource_name": r.resource_name,
                "resource_type": r.resource_type,
                "hospital_id": r.hospital_id,
                "hospital_name": r.hospital_name,
                "route_id": r.route_id,
                "trust_score": r.trust_score,
                "score_breakdown": r.score_breakdown,
                "explanation": r.explanation,
                "approval_status": r.approval_status,
                "ambulance_coords": r.ambulance_coords,
                "hospital_coords": r.hospital_coords,
                "route_geometry": r.route_geometry
            })
            
        updates.append({
            "incident_id": incident.id,
            "recommendations": rec_data
        })
        
        # Log event
        event = EmergencyEvent(
            incident_id=incident.id,
            event_type="Re_plan_Triggered",
            event_metadata={"change": change_description}
        )
        db.add(event)
        db.commit()

    # Broadcast to all WebSocket listeners
    await ws_manager.broadcast({
        "type": "DYNAMIC_REPLAN",
        "description": change_description,
        "updates": updates
    })
