import os
import sys
import asyncio
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ARRAY
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import List, Dict, Any

# Adjust path to import backend modules locally
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our backend components
from app.routing import calculate_route, add_blockage, remove_blockage, get_active_blockages
from app.scoring import calculate_trust_score
from app.ai_service import analyze_emergency_report
from app.rag_service import fit_vectorizer_on_guidelines, query_rag

# Setup SQLite in-memory DB for tests
Base = declarative_base()

class TestUser(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    role = Column(String)

class TestIncident(Base):
    __tablename__ = "incidents"
    id = Column(Integer, primary_key=True)
    incident_type = Column(String)
    severity = Column(String)
    description = Column(Text)
    latitude = Column(Float)
    longitude = Column(Float)
    status = Column(String)

class TestResource(Base):
    __tablename__ = "resources"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    type = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    status = Column(String)
    availability = Column(Boolean)

class TestHospital(Base):
    __tablename__ = "hospitals"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    emergency_capacity = Column(Integer)
    current_occupancy = Column(Integer)
    specialties = Column(String) # SQLite doesn't support ARRAY, we will mock specialties as comma-separated string
    availability = Column(Boolean)

class TestRecommendation(Base):
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True)
    incident_id = Column(Integer)
    resource_id = Column(Integer)
    hospital_id = Column(Integer)
    trust_score = Column(Float)
    explanation = Column(Text)

async def run_local_tests():
    print("==================================================")
    print("          ResQMesh Business Logic Verification")
    print("==================================================")

    # Set up credentials temporarily for the test run
    os.environ["OPENAI_API_BASE"] = "https://api.17.wtf/v1"
    os.environ["OPENAI_API_KEY"] = "sk-lm0-f57ccf3636a54ce1a6a462fb3a96044f"
    os.environ["OPENAI_MODEL_NAME"] = "posiden/nemotron-3-ultra"

    # 1. Test AI Report Understanding & Extraction
    print("\n--- 1. Testing AI Incident Intelligence ---")
    emergency_text = "A major multi-vehicle car collision has occurred on Market St. 3 people are injured and bleeding. Smoke is rising."
    print(f"Citizen Report: \"{emergency_text}\"")
    
    # Analyze report
    ai_result = await analyze_emergency_report(emergency_text)
    print(f"AI Detected Type:      {ai_result.incident_type}")
    print(f"AI Detected Severity:  {ai_result.severity}")
    print(f"Estimated Victims:     {ai_result.num_victims}")
    print(f"Required Services:     {ai_result.required_services}")
    print(f"Extracted Key Details: {ai_result.key_details}")
    
    assert ai_result.incident_type in ["Car Collision", "Structure Fire", "General Medical"], "AI Incident Type extraction error"
    assert ai_result.severity in ["High", "Critical", "Medium"], "AI Severity assessment error"
    print("✓ Incident Intelligence verified successfully.")

    # 2. Test RAG Retrieval Logic
    print("\n--- 2. Testing RAG Guideline Search ---")
    fit_vectorizer_on_guidelines("data/emergency_guidance.txt")
    # Query guidelines locally (using our optimized sklearn fallback when pgvector is not initialized)
    guidance = query_rag(None, emergency_text)
    print(f"Retrieved Procedure:\n{guidance}")
    assert "Motor Vehicle Accident" in guidance or "Bleeding" in guidance or "Fire" in guidance, "RAG Search failed to find relevant documentation"
    print("✓ pgvector RAG fallback semantic retrieval verified successfully.")

    # 3. Test Routing and TrustScore Engine
    print("\n--- 3. Testing Routing & TrustScore Calculation ---")
    
    # Setup mock incident location (Civic Center SF)
    # Mock Incident
    class MockIncident:
        latitude = 37.7794
        longitude = -122.4194
        incident_type = "Car Collision"
        severity = "High"
        description = emergency_text

    # Mock Resource (Ambulance 1: Union Square - 1.5 km away)
    class MockAmbulance1:
        id = 1
        name = "Medic-01 (Union Square)"
        type = "Ambulance"
        latitude = 37.7892
        longitude = -122.4012
        status = "Idle"
        availability = True

    # Mock Resource (Ambulance 2: Nob Hill - 2.8 km away)
    class MockAmbulance2:
        id = 2
        name = "Medic-02 (Nob Hill)"
        type = "Ambulance"
        latitude = 37.7951
        longitude = -122.4182
        status = "Idle"
        availability = True

    # Mock Hospital (ZSFG - Trauma Specialty, 2.5 km away, capacity available)
    class MockHospital1:
        id = 1
        name = "Zuckerberg SF General Hospital"
        latitude = 37.7568
        longitude = -122.4058
        emergency_capacity = 50
        current_occupancy = 30
        specialties = ["Trauma", "Cardiac"]
        availability = True

    # Mock Hospital 2 (Kaiser SF - No Trauma specialty, 3.2 km away)
    class MockHospital2:
        id = 2
        name = "Kaiser Permanente SF"
        latitude = 37.7831
        longitude = -122.4431
        emergency_capacity = 25
        current_occupancy = 24 # Almost Full
        specialties = ["Cardiac", "Pediatric"]
        availability = True

    inc = MockIncident()
    amb1 = MockAmbulance1()
    amb2 = MockAmbulance2()
    hosp1 = MockHospital1()
    hosp2 = MockHospital2()

    # Calculate TrustScores under normal conditions
    score_option1 = await calculate_trust_score(inc, amb1, hosp1)
    score_option2 = await calculate_trust_score(inc, amb2, hosp2)

    print(f"Option 1: Dispatches {amb1.name} to {hosp1.name}")
    print(f"  TrustScore: {score_option1['trust_score']} / 100")
    print(f"  Details: {score_option1['explanation']}")
    
    print(f"Option 2: Dispatches {amb2.name} to {hosp2.name}")
    print(f"  TrustScore: {score_option2['trust_score']} / 100")
    print(f"  Details: {score_option2['explanation']}")

    assert score_option1["trust_score"] > score_option2["trust_score"], "TrustScore failed to rank ZSFG trauma center and closer ambulance first!"
    print("✓ TrustScore ranking matches operational guidelines.")

    # 4. Test Dynamic Re-planning on road blockages
    print("\n--- 4. Testing Dynamic Re-planning Engine ---")
    print("Simulating a major road block on Market St crossing downtown Union Square streets (placing blockage near Union Square)...")
    
    # Place a blockage near Union Square (lat 37.788, lon -122.408) which blocks Medic-01's route to the incident
    add_blockage("BLOCK_UNION_SQ", 37.788, -122.408, "Major Water Main Break on Stockton St", radius_m=200.0)
    
    # Recalculate route and TrustScores with blockage active
    score_option1_blocked = await calculate_trust_score(inc, amb1, hosp1)
    score_option2_blocked = await calculate_trust_score(inc, amb2, hosp1) # Amb 2 to Hosp 1

    print(f"Blocked Option 1: Dispatches {amb1.name} to {hosp1.name}")
    print(f"  New TrustScore: {score_option1_blocked['trust_score']} / 100")
    print(f"  Details: {score_option1_blocked['explanation']}")

    print(f"Blocked Option 2: Dispatches {amb2.name} to {hosp1.name}")
    print(f"  New TrustScore: {score_option2_blocked['trust_score']} / 100")
    print(f"  Details: {score_option2_blocked['explanation']}")

    assert score_option1_blocked["trust_score"] == 0.0, "Dynamic re-planning failed to set blocked route score to 0!"
    assert score_option2_blocked["trust_score"] > 0.0, "Unblocked alternate route score error!"
    print("✓ Dynamic Re-planning recalculated successfully (Option 2 is now the primary recommendation).")

    # Clear blockage
    remove_blockage("BLOCK_UNION_SQ")
    print("Cleared road blockage.")
    
    print("\n==================================================")
    print("  ALL RESQMESH CORE LOGIC VERIFICATION CHECKS PASSED")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_local_tests())
