import os
import json
import re
from openai import AsyncOpenAI
from .schemas import AIStructuredIncident

# Initialize OpenAI Client pointing to user's endpoint
api_key = os.getenv("OPENAI_API_KEY", "sk-lm0-f57ccf3636a54ce1a6a462fb3a96044f")
api_base = os.getenv("OPENAI_API_BASE", "https://api.17.wtf/v1")
model_name = os.getenv("OPENAI_MODEL_NAME", "posiden/nemotron-3-ultra")

client = None
if api_key:
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base
    )

def rule_based_fallback(description: str) -> AIStructuredIncident:
    """
    Keyword-based emergency parser as a robust fallback.
    """
    desc_lower = description.lower()
    
    # Defaults
    inc_type = "General Medical"
    severity = "Medium"
    num_victims = 1
    services = ["EMS"]
    details = []

    # 1. Classification rules
    if any(k in desc_lower for k in ["fire", "smoke", "burn", "explosion"]):
        inc_type = "Structure Fire"
        severity = "High"
        services = ["Fire", "EMS"]
        details.append("Possible active fire or thermal hazard.")
    elif any(k in desc_lower for k in ["crash", "accident", "collision", "car", "truck"]):
        inc_type = "Car Collision"
        severity = "High"
        services = ["EMS", "Police", "Fire"]
        details.append("Motor vehicle accident reported.")
    elif any(k in desc_lower for k in ["chest pain", "heart", "cardiac", "stroke", "unresponsive", "passed out", "unconscious"]):
        inc_type = "Cardiac Arrest"
        severity = "Critical"
        services = ["EMS"]
        details.append("Patient has cardiac/unconscious indicators.")
    elif any(k in desc_lower for k in ["shoot", "gun", "bullet", "wound", "bleed", "stab"]):
        inc_type = "Gunshot Wound"
        severity = "Critical"
        services = ["EMS", "Police"]
        details.append("Trauma/violence-related injury.")
    elif any(k in desc_lower for k in ["leak", "gas", "chemical", "fume", "poison"]):
        inc_type = "Gas Leak"
        severity = "High"
        services = ["Fire", "Hazmat"]
        details.append("Inhalation hazard/hazardous material.")
    elif any(k in desc_lower for k in ["seizure", "epilepsy"]):
        inc_type = "Seizure"
        severity = "High"
        services = ["EMS"]
        details.append("Active seizure activity.")

    # Severity overrides
    if "critical" in desc_lower or "dying" in desc_lower or "bleeding heavily" in desc_lower or "not breathing" in desc_lower:
        severity = "Critical"
    elif "minor" in desc_lower or "scratches" in desc_lower or "cold" in desc_lower:
        severity = "Low"

    # Victim count estimation
    match = re.search(r"(\d+)\s+(people|person|victims|injured)", desc_lower)
    if match:
        num_victims = int(match.group(1))
    
    details.append("Parsed via local rule-based engine (AI API was offline/fallback).")
    
    return AIStructuredIncident(
        incident_type=inc_type,
        severity=severity,
        num_victims=num_victims,
        required_services=services,
        key_details=details
    )

async def analyze_emergency_report(description: str) -> AIStructuredIncident:
    """
    Analyzes emergency text using user-provided OpenAI-compatible API.
    Falls back to a keyword-based matcher if the API call fails.
    """
    if not client:
        return rule_based_fallback(description)

    prompt = f"""
    You are an expert emergency dispatch parser. Your task is to analyze the following unstructured report and extract structured details.
    
    Emergency Report:
    "{description}"
    
    You MUST output valid JSON strictly matching the following schema:
    {{
      "incident_type": "string (e.g. Cardiac Arrest, Structure Fire, Car Collision, Gunshot Wound, Gas Leak, General Medical, Seizure)",
      "severity": "string (strictly one of: Low, Medium, High, Critical)",
      "num_victims": "integer (estimated count, default is 1 if unspecified)",
      "required_services": ["array of strings", "e.g. EMS, Fire, Police, Hazmat"],
      "key_details": ["list of strings detailing critical observations"]
    }}
    
    Return ONLY the raw JSON block without markdown formatting or code blocks.
    """
    
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a professional emergency responder dispatch AI that only outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500,
            timeout=2.0
        )
        
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response content from AI completions")
        content = content.strip()
        
        # Clean markdown code block if present
        if content.startswith("```"):
            # Strip first line
            content = re.sub(r"^```[a-zA-Z0-9]*\n", "", content)
            # Strip last line
            content = re.sub(r"\n```$", "", content)
            content = content.strip()

        # Parse JSON
        data = json.loads(content)
        return AIStructuredIncident(
            incident_type=data.get("incident_type", "General Medical"),
            severity=data.get("severity", "Medium"),
            num_victims=int(data.get("num_victims", 1)),
            required_services=data.get("required_services", ["EMS"]),
            key_details=data.get("key_details", ["Extracted details from AI report analysis."])
        )
    except Exception as e:
        print(f"AI Service error, using rule-based fallback: {e}")
        return rule_based_fallback(description)

async def explain_recommendation(
    incident_desc: str,
    incident_type: str,
    ambulance_name: str,
    hospital_name: str,
    eta_mins: float
) -> str:
    """
    Queries the custom LLM to provide a clear, supportive operator explanation.
    """
    if not client:
        return f"Dispatched {ambulance_name} to respond to the incident, transferring the patient to {hospital_name} (ETA: {round(eta_mins, 1)}m)."

    prompt = f"""
    Explain to an emergency operator why this choice of resource and hospital is recommended:
    - Incident: {incident_desc} (Type: {incident_type})
    - Dispatching: {ambulance_name}
    - Transporting to: {hospital_name}
    - Combined Response & Transport ETA: {round(eta_mins, 1)} minutes.
    
    Keep the explanation to 2 sentences. Be concise, clear, and highlight capacity or specialty alignment.
    """
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a brief, helpful emergency response assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150,
            timeout=2.0
        )
        content = response.choices[0].message.content
        if content:
            return content.strip()
        else:
            raise ValueError("Empty explanation content from AI completions")
    except Exception as e:
        print(f"AI explanation error: {e}")
        return f"Dispatched {ambulance_name} to respond to the incident, transferring the patient to {hospital_name} (ETA: {round(eta_mins, 1)}m)."
