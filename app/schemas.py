from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# User Schemas
class UserBase(BaseModel):
    name: str
    email: str
    role: str

class UserCreate(UserBase):
    pass

class UserResponse(UserBase):
    id: int
    created_at: datetime
    class Config:
        from_attributes = True

# Incident Schemas
class IncidentBase(BaseModel):
    incident_type: str
    severity: str
    description: str
    latitude: float
    longitude: float
    status: str

class IncidentCreate(BaseModel):
    description: str
    latitude: float
    longitude: float
    user_id: Optional[int] = None

class IncidentResponse(IncidentBase):
    id: int
    user_id: Optional[int]
    created_at: datetime
    class Config:
        from_attributes = True

# Resource Schemas
class ResourceBase(BaseModel):
    name: str
    type: str
    latitude: float
    longitude: float
    status: str
    availability: bool

class ResourceUpdate(BaseModel):
    status: Optional[str] = None
    availability: Optional[bool] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class ResourceResponse(ResourceBase):
    id: int
    created_at: datetime
    class Config:
        from_attributes = True

# Hospital Schemas
class HospitalBase(BaseModel):
    name: str
    latitude: float
    longitude: float
    emergency_capacity: int
    current_occupancy: int
    specialties: List[str]
    availability: bool

class HospitalUpdate(BaseModel):
    current_occupancy: Optional[int] = None
    availability: Optional[bool] = None

class HospitalResponse(HospitalBase):
    id: int
    created_at: datetime
    class Config:
        from_attributes = True

# Recommendation Schemas
class TrustScoreBreakdown(BaseModel):
    distance_score: float
    time_score: float
    capacity_score: float
    specialty_score: float
    route_condition_score: float
    total_score: float
    duration_sec: Optional[float] = None

class RecommendationResponse(BaseModel):
    id: int
    incident_id: int
    resource_id: Optional[int]
    resource_name: Optional[str]
    resource_type: Optional[str]
    hospital_id: Optional[int]
    hospital_name: Optional[str]
    route_id: Optional[str]
    trust_score: float
    score_breakdown: TrustScoreBreakdown
    explanation: str
    approval_status: str
    created_at: datetime
    
    # Coordinates for mapping
    ambulance_coords: Optional[List[float]] = None # [lat, lon]
    hospital_coords: Optional[List[float]] = None  # [lat, lon]
    route_geometry: Optional[List[List[float]]] = None # List of [lat, lon] points
    
    class Config:
        from_attributes = True

class EmergencyEventResponse(BaseModel):
    id: int
    incident_id: int
    event_type: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = Field(None, validation_alias="event_metadata", serialization_alias="metadata")
    class Config:
        from_attributes = True
        populate_by_name = True

# AI Service Structured Extractor Schema
class AIStructuredIncident(BaseModel):
    incident_type: str = Field(description="The classified type of emergency, e.g., 'Cardiac Arrest', 'Structure Fire', 'Car Collision', 'Gunshot Wound', 'Gas Leak', 'General Medical', 'Assault'")
    severity: str = Field(description="Severity level: 'Low', 'Medium', 'High', or 'Critical'")
    num_victims: int = Field(description="Extracted or estimated number of victims. Default is 0 if unspecified.")
    required_services: List[str] = Field(description="List of services needed, e.g., ['EMS', 'Fire', 'Police', 'Hazmat']")
    key_details: List[str] = Field(description="Bullet points of critical medical/operational observations extracted from description.")

# Live Report Processing Response
class IncidentReportResult(BaseModel):
    incident: IncidentResponse
    structured_data: AIStructuredIncident
    guidance: str
    recommendations: List[RecommendationResponse]

# Simulated blockage schema
class RoadBlockage(BaseModel):
    id: str
    latitude: float
    longitude: float
    description: str
    radius_meters: float = 150.0 # radius of influence
