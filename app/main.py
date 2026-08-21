import os
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from .database import engine, Base, SessionLocal, get_db
from .models import Incident, Resource, Hospital, Recommendation, EmergencyEvent
from .schemas import (
    IncidentCreate, IncidentResponse, IncidentReportResult,
    ResourceResponse, ResourceUpdate, HospitalResponse, HospitalUpdate,
    RecommendationResponse, EmergencyEventResponse, RoadBlockage
)
from .ai_service import analyze_emergency_report
from .rag_service import initialize_rag, query_rag
from .scoring import calculate_trust_score
from .routing import calculate_route, add_blockage, remove_blockage, get_active_blockages
from .dynamic_replanning import ws_manager, recalculate_recommendations, trigger_dynamic_replanning

app = FastAPI(title="ResQMesh Emergency Response Backend")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, lock this down
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "ResQMesh Emergency Response Backend",
        "docs": "/docs",
        "api": {
            "incidents": "/api/incidents",
            "resources": "/api/resources",
            "hospitals": "/api/hospitals"
        }
    }

from .models import User, Hospital, Resource

def seed_database(db: Session):
    # Check if Users exists
    if db.query(User).count() == 0:
        print("Seeding SQLite database with default mock data...")
        users = [
            User(name="System Operator", email="operator@resqmesh.gov", role="operator"),
            User(name="System Administrator", email="admin@resqmesh.gov", role="admin"),
            User(name="John Doe", email="john.doe@gmail.com", role="citizen")
        ]
        db.add_all(users)
        db.commit()

    if db.query(Hospital).count() == 0:
        hospitals = [
            Hospital(name="SMS Hospital (Sawai Man Singh Hospital)", latitude=26.8982, longitude=75.8124, emergency_capacity=50, current_occupancy=42, specialties=["Trauma", "Cardiac"], availability=True),
            Hospital(name="Fortis Escorts Hospital", latitude=26.8488, longitude=75.8015, emergency_capacity=35, current_occupancy=18, specialties=["Trauma", "Pediatric", "Burn"], availability=True),
            Hospital(name="Eternal Hospital (EHCC)", latitude=26.8542, longitude=75.8066, emergency_capacity=25, current_occupancy=20, specialties=["Cardiac", "Pediatric"], availability=True),
            Hospital(name="Santokba Durlabhji Hospital (SDMH)", latitude=26.8943, longitude=75.8037, emergency_capacity=30, current_occupancy=24, specialties=["Trauma", "Burn"], availability=True)
        ]
        db.add_all(hospitals)
        db.commit()

    if db.query(Resource).count() == 0:
        resources = [
            Resource(name="Medic-01 (C-Scheme)", type="Ambulance", latitude=26.9094, longitude=75.8012, status="Idle", availability=True),
            Resource(name="Medic-02 (Malviya Nagar)", type="Ambulance", latitude=26.8548, longitude=75.8214, status="Idle", availability=True),
            Resource(name="Medic-03 (Vaishali Nagar)", type="Ambulance", latitude=26.9015, longitude=75.7382, status="Idle", availability=True),
            Resource(name="Medic-04 (Raja Park)", type="Ambulance", latitude=26.8912, longitude=75.8294, status="Busy", availability=False),
            Resource(name="Rescue-10 (Mansarovar)", type="FireTruck", latitude=26.8621, longitude=75.7562, status="Idle", availability=True),
            Resource(name="Patrol-22 (Pink City)", type="PoliceCruiser", latitude=26.9214, longitude=75.8252, status="Idle", availability=True)
        ]
        db.add_all(resources)
        db.commit()

# Startup Hook: Ensure database tables are created & RAG populated
@app.on_event("startup")
def startup_event():
    print("Initializing Database tables...")
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Clear old database records for fresh simulation state
        print("Clearing old simulation records (incidents, recommendations, events, resources, hospitals)...")
        db.query(Hospital).delete()
        db.query(Resource).delete()
        db.query(Recommendation).delete()
        db.query(EmergencyEvent).delete()
        db.query(Incident).delete()
        db.commit()

        seed_database(db)
        
        # Reset hospital occupancy to defaults
        for h in db.query(Hospital).all():
            if "SMS" in h.name or "Sawai" in h.name:
                h.current_occupancy = 42
            elif "Fortis" in h.name:
                h.current_occupancy = 18
            elif "EHCC" in h.name or "Eternal" in h.name:
                h.current_occupancy = 20
            elif "SDMH" in h.name or "Santokba" in h.name:
                h.current_occupancy = 24
                
        # Reset resource status to defaults
        for r in db.query(Resource).all():
            if "Medic-04" in r.name:
                r.status = "Busy"
                r.availability = False
            else:
                r.status = "Idle"
                r.availability = True
                
        db.commit()
        
        print("Initializing RAG database...")
        initialize_rag(db)
    finally:
        db.close()
    print("Backend Startup complete.")

# ----------------- Incident Management Endpoints -----------------

@app.post("/api/incidents/report", response_model=IncidentReportResult)
async def report_incident(payload: IncidentCreate, db: Session = Depends(get_db)):
    """
    1. Citizens report emergency with text and location.
    2. AI classifies incident type, severity, and services needed.
    3. RAG retrieves first-aid guidance.
    4. TrustScore engine generates recommendations.
    5. Real-time broadcast pushes new incident to operators.
    """
    # 1. Create Incident
    incident = Incident(
        user_id=payload.user_id,
        description=payload.description,
        latitude=payload.latitude,
        longitude=payload.longitude,
        incident_type="Processing...", # temp placeholder
        severity="Medium", # temp placeholder
        status="Pending"
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)

    # 2. Run AI Incident Intelligence (Structured Output Extraction)
    ai_structured = await analyze_emergency_report(payload.description)
    
    # Update incident with extracted facts
    incident.incident_type = ai_structured.incident_type
    incident.severity = ai_structured.severity
    db.commit()
    db.refresh(incident)

    # 3. Query RAG system for emergency instructions
    guidance = query_rag(db, payload.description)

    # 4. Log reporting event
    event = EmergencyEvent(
        incident_id=incident.id,
        event_type="Incident_Reported",
        event_metadata={
            "description": payload.description,
            "ai_extracted": ai_structured.model_dump()
        }
    )
    db.add(event)
    db.commit()

    # 5. Run TrustScore engine to generate recommendations
    recs = await recalculate_recommendations(db, incident.id)

    # Format recommendations for return schema
    rec_responses = []
    for r in recs:
        rec_responses.append(
            RecommendationResponse(
                id=r.id,
                incident_id=r.incident_id,
                resource_id=r.resource_id,
                resource_name=r.resource_name,
                resource_type=r.resource_type,
                hospital_id=r.hospital_id,
                hospital_name=r.hospital_name,
                route_id=r.route_id,
                trust_score=r.trust_score,
                score_breakdown=r.score_breakdown,
                explanation=r.explanation,
                approval_status=r.approval_status,
                ambulance_coords=r.ambulance_coords,
                hospital_coords=r.hospital_coords,
                route_geometry=r.route_geometry,
                created_at=r.created_at
            )
        )

    # 6. Broadcast updated state to Command Center WebSockets
    incident_resp = IncidentResponse.model_validate(incident)
    await ws_manager.broadcast({
        "type": "NEW_INCIDENT",
        "incident": incident_resp.model_dump(mode="json"),
        "structured_data": ai_structured.model_dump(mode="json"),
        "guidance": guidance,
        "recommendations": [r.model_dump(mode="json") for r in rec_responses]
    })

    return IncidentReportResult(
        incident=incident_resp,
        structured_data=ai_structured,
        guidance=guidance,
        recommendations=rec_responses
    )

@app.get("/api/incidents", response_model=List[IncidentResponse])
def get_incidents(db: Session = Depends(get_db)):
    return db.query(Incident).order_by(Incident.created_at.desc()).all()

@app.get("/api/incidents/{incident_id}", response_model=IncidentResponse)
def get_incident(incident_id: int, db: Session = Depends(get_db)):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident

@app.get("/api/incidents/{incident_id}/recommendations", response_model=List[RecommendationResponse])
async def get_incident_recommendations(incident_id: int, db: Session = Depends(get_db)):
    """
    Returns recommendations with spatial coordinates populated.
    """
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
        
    recs = db.query(Recommendation).filter(Recommendation.incident_id == incident_id).all()
    
    # If recommendations aren't generated yet (e.g. empty DB), generate them on the fly
    if not recs:
        recs = await recalculate_recommendations(db, incident_id)
        
    response_recs = []
    for r in recs:
        # Resolve spatial routing details
        res = db.query(Resource).filter(Resource.id == r.resource_id).first()
        hosp = db.query(Hospital).filter(Hospital.id == r.hospital_id).first()
        
        # Calculate routes dynamically for geometry output
        if res and hosp:
            score_data = await calculate_trust_score(incident, res, hosp)
            score_data["score_breakdown"]["duration_sec"] = score_data["duration_sec"]
            ambulance_coords = score_data["ambulance_coords"]
            hospital_coords = score_data["hospital_coords"]
            route_geometry = score_data["route_geometry"]
            score_breakdown = score_data["score_breakdown"]
        else:
            ambulance_coords = None
            hospital_coords = None
            route_geometry = None
            score_breakdown = r.score_breakdown

        response_recs.append(
            RecommendationResponse(
                id=r.id,
                incident_id=r.incident_id,
                resource_id=r.resource_id,
                resource_name=res.name if res else None,
                resource_type=res.type if res else None,
                hospital_id=r.hospital_id,
                hospital_name=hosp.name if hosp else None,
                route_id=r.route_id,
                trust_score=r.trust_score,
                score_breakdown=score_breakdown,
                explanation=r.explanation,
                approval_status=r.approval_status,
                ambulance_coords=ambulance_coords,
                hospital_coords=hospital_coords,
                route_geometry=route_geometry,
                created_at=r.created_at
            )
        )
        
    # Sort by trust score
    response_recs.sort(key=lambda x: x.trust_score, reverse=True)
    return response_recs

@app.get("/api/incidents/{incident_id}/events", response_model=List[EmergencyEventResponse])
def get_incident_events(incident_id: int, db: Session = Depends(get_db)):
    return db.query(EmergencyEvent).filter(EmergencyEvent.incident_id == incident_id).order_by(EmergencyEvent.timestamp.desc()).all()

# ----------------- Resource Management Endpoints -----------------

@app.get("/api/resources", response_model=List[ResourceResponse])
def get_resources(db: Session = Depends(get_db)):
    return db.query(Resource).all()

@app.put("/api/resources/{resource_id}", response_model=ResourceResponse)
async def update_resource(resource_id: int, payload: ResourceUpdate, db: Session = Depends(get_db)):
    resource = db.query(Resource).filter(Resource.id == resource_id).first()
    if not resource:
        raise HTTPException(status_code=404, detail="Resource not found")
        
    for var, val in payload.model_dump(exclude_unset=True).items():
        setattr(resource, var, val)
    db.commit()
    db.refresh(resource)
    
    # Trigger Dynamic Re-planning on resource change
    change_desc = f"Ambulance {resource.name} status updated to {resource.status} (Availability: {resource.availability})"
    await trigger_dynamic_replanning(db, change_desc)
    
    return resource

# ----------------- Hospital Management Endpoints -----------------

@app.get("/api/hospitals", response_model=List[HospitalResponse])
def get_hospitals(db: Session = Depends(get_db)):
    return db.query(Hospital).all()

@app.put("/api/hospitals/{hospital_id}", response_model=HospitalResponse)
async def update_hospital(hospital_id: int, payload: HospitalUpdate, db: Session = Depends(get_db)):
    hospital = db.query(Hospital).filter(Hospital.id == hospital_id).first()
    if not hospital:
        raise HTTPException(status_code=404, detail="Hospital not found")
        
    for var, val in payload.model_dump(exclude_unset=True).items():
        setattr(hospital, var, val)
    db.commit()
    db.refresh(hospital)
    
    # Trigger Dynamic Re-planning on hospital change
    change_desc = f"Hospital {hospital.name} updated (Capacity occupancy: {hospital.current_occupancy})"
    await trigger_dynamic_replanning(db, change_desc)
    
    return hospital

# ----------------- Recommendation / Approval Endpoints -----------------

@app.post("/api/recommendations/{rec_id}/approve", response_model=RecommendationResponse)
async def approve_recommendation(rec_id: int, db: Session = Depends(get_db)):
    """
    Approves the recommended dispatch and updates incident and resource statuses.
    """
    rec = db.query(Recommendation).filter(Recommendation.id == rec_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Recommendation not found")
        
    incident = db.query(Incident).filter(Incident.id == rec.incident_id).first()
    resource = db.query(Resource).filter(Resource.id == rec.resource_id).first()
    hospital = db.query(Hospital).filter(Hospital.id == rec.hospital_id).first()
    
    # Update recommendation approval status
    rec.approval_status = "Approved"
    
    # Reject other recommendations for this incident
    other_recs = db.query(Recommendation).filter(
        Recommendation.incident_id == rec.incident_id,
        Recommendation.id != rec.id
    ).all()
    for o_rec in other_recs:
        o_rec.approval_status = "Rejected"
        
    # Update Incident status
    incident.status = "Dispatched"
    
    # Update Ambulance availability and status
    if resource:
        resource.status = "EnRoute"
        resource.availability = False
        
    # Update Hospital current occupancy
    if hospital:
        hospital.current_occupancy = min(hospital.emergency_capacity, hospital.current_occupancy + 1)
        
    # Log approval event
    event = EmergencyEvent(
        incident_id=incident.id,
        event_type="Operator_Approved",
        event_metadata={
            "approved_recommendation_id": rec_id,
            "dispatched_ambulance": resource.name if resource else None,
            "destination_hospital": hospital.name if hospital else None
        }
    )
    db.add(event)
    db.commit()
    db.refresh(rec)
    
    # Broadcast status change to operators
    await ws_manager.broadcast({
        "type": "INCIDENT_DISPATCHED",
        "incident_id": incident.id,
        "status": "Dispatched",
        "resource_name": resource.name if resource else None,
        "hospital_name": hospital.name if hospital else None
    })
    
    # Mock dynamic coordinates for returning response
    rec.resource_name = resource.name if resource else None
    rec.resource_type = resource.type if resource else None
    rec.hospital_name = hospital.name if hospital else None
    
    return rec

@app.post("/api/recommendations/{rec_id}/reject", response_model=RecommendationResponse)
def reject_recommendation(rec_id: int, db: Session = Depends(get_db)):
    rec = db.query(Recommendation).filter(Recommendation.id == rec_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Recommendation not found")
        
    rec.approval_status = "Rejected"
    
    event = EmergencyEvent(
        incident_id=rec.incident_id,
        event_type="Operator_Rejected",
        event_metadata={"rejected_recommendation_id": rec_id}
    )
    db.add(event)
    db.commit()
    db.refresh(rec)
    
    return rec

# ----------------- Road Blockage Management Endpoints -----------------

@app.post("/api/blockages", response_model=RoadBlockage)
async def create_blockage(blockage: RoadBlockage, db: Session = Depends(get_db)):
    """
    Simulates placing a new road block. Recalculates all recommendations.
    """
    add_blockage(
        blockage_id=blockage.id,
        lat=blockage.latitude,
        lon=blockage.longitude,
        description=blockage.description,
        radius_m=blockage.radius_meters
    )
    # Trigger Dynamic Re-planning on route change
    await trigger_dynamic_replanning(db, f"New road blockage registered: {blockage.description}")
    
    return blockage

@app.get("/api/blockages", response_model=List[RoadBlockage])
def get_blockages():
    blocks = get_active_blockages()
    return [
        RoadBlockage(
            id=b["id"],
            latitude=b["latitude"],
            longitude=b["longitude"],
            description=b["description"],
            radius_meters=b["radius_meters"]
        ) for b in blocks
    ]

@app.delete("/api/blockages/{blockage_id}")
async def delete_blockage(blockage_id: str, db: Session = Depends(get_db)):
    remove_blockage(blockage_id)
    # Trigger Dynamic Re-planning on route change
    await trigger_dynamic_replanning(db, f"Road blockage cleared: {blockage_id}")
    return {"status": "success", "message": f"Blockage {blockage_id} removed"}

# ----------------- WebSocket Communication Hub -----------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Maintain connection and listen for heartbeat
            data = await websocket.receive_text()
            # Simple heartbeat ping/pong
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket connection error: {e}")
        ws_manager.disconnect(websocket)
