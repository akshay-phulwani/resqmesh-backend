from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False, index=True)
    role = Column(String(20), nullable=False)  # 'citizen', 'operator', 'admin'
    created_at = Column(DateTime, server_default=func.now())

class Incident(Base):
    __tablename__ = "incidents"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    incident_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)  # 'Low', 'Medium', 'High', 'Critical'
    description = Column(Text, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    status = Column(String(20), nullable=False, default="Pending")  # 'Pending', 'Dispatched', 'Resolved', 'Cancelled'
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    user = relationship("User")
    recommendations = relationship("Recommendation", back_populates="incident", cascade="all, delete-orphan")

class Resource(Base):
    __tablename__ = "resources"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    type = Column(String(20), nullable=False)  # 'Ambulance', 'FireTruck', 'PoliceCruiser'
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    status = Column(String(20), nullable=False, default="Idle")  # 'Idle', 'EnRoute', 'Busy', 'Offline'
    availability = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())

class Hospital(Base):
    __tablename__ = "hospitals"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    emergency_capacity = Column(Integer, nullable=False)
    current_occupancy = Column(Integer, default=0)
    specialties = Column(JSON, nullable=False)
    availability = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())

class Recommendation(Base):
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False)
    resource_id = Column(Integer, ForeignKey("resources.id", ondelete="SET NULL"), nullable=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id", ondelete="SET NULL"), nullable=True)
    route_id = Column(String(100), nullable=True)
    trust_score = Column(Float, nullable=False)
    score_breakdown = Column(JSON, nullable=False)  # details of score
    explanation = Column(Text, nullable=False)
    approval_status = Column(String(20), nullable=False, default="Pending")  # 'Pending', 'Approved', 'Rejected'
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    incident = relationship("Incident", back_populates="recommendations")
    resource = relationship("Resource")
    hospital = relationship("Hospital")

class EmergencyEvent(Base):
    __tablename__ = "emergency_events"
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False)
    event_type = Column(String(50), nullable=False)
    timestamp = Column(DateTime, server_default=func.now())
    event_metadata = Column("metadata", JSON, nullable=True)

class GuidanceEmbedding(Base):
    __tablename__ = "guidance_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=True)  # JSON-serialized list of 384 float dimensions
