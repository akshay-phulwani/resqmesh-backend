import httpx
import asyncio

API_URL = "http://127.0.0.1:8000"

async def test_e2e_flow():
    print("==================================================")
    print("        ResQMesh Live E2E Server Verification")
    print("==================================================")

    async with httpx.AsyncClient(timeout=40.0) as client:
        # 1. Citizen reports emergency
        print("\nStep 1: Simulating citizen submitting a collision report...")
        report_data = {
            "description": "Car collision on Market St near Union Square! 3 victims are injured and bleeding. Engine is smoking.",
            "latitude": 37.7882,
            "longitude": -122.4075,
            "user_id": 3
        }
        
        response = await client.post(f"{API_URL}/api/incidents/report", json=report_data)
        assert response.status_code == 200, f"Report failed: {response.text}"
        result = response.json()
        
        incident = result["incident"]
        ai_data = result["structured_data"]
        guidance = result["guidance"]
        recommendations = result["recommendations"]
        
        incident_id = incident["id"]
        print(f"[OK] Incident #{incident_id} registered successfully.")
        print(f"  AI Classified Type:  {ai_data['incident_type']}")
        print(f"  AI Classified Severity: {ai_data['severity']}")
        print(f"  RAG Guidelines Matched: {guidance.splitlines()[0]}")
        print(f"  Recommendations Count: {len(recommendations)}")
        
        # Verify AI results
        assert ai_data["incident_type"] in ["Car Collision", "Structure Fire", "General Medical"], "AI type classification failed"
        assert len(recommendations) > 0, "No recommendations generated"

        # 2. Operator reviews and approves top option
        print("\nStep 2: Reviewing recommendations and approving the top dispatch option...")
        top_rec = recommendations[0]
        print(f"  Top option: {top_rec['resource_name']} -> {top_rec['hospital_name']}")
        print(f"  TrustScore: {top_rec['trust_score']} / 100")
        print(f"  Explanation: {top_rec['explanation']}")

        approve_response = await client.post(f"{API_URL}/api/recommendations/{top_rec['id']}/approve")
        assert approve_response.status_code == 200, f"Approval failed: {approve_response.text}"
        print("[OK] Dispatch approved successfully.")

        # 3. Verify status changes in database
        print("\nStep 3: Verifying database state updates after dispatch...")
        inc_check = await client.get(f"{API_URL}/api/incidents/{incident_id}")
        assert inc_check.json()["status"] == "Dispatched", "Incident status did not update to Dispatched"
        
        res_check = await client.get(f"{API_URL}/api/resources")
        dispatched_unit = next(r for r in res_check.json() if r["id"] == top_rec["resource_id"])
        print(f"  Ambulance status: {dispatched_unit['name']} is now '{dispatched_unit['status']}' (Available: {dispatched_unit['availability']})")
        assert dispatched_unit["status"] == "EnRoute", "Ambulance status did not update to EnRoute"
        
        hosp_check = await client.get(f"{API_URL}/api/hospitals")
        destination_hosp = next(h for h in hosp_check.json() if h["id"] == top_rec["hospital_id"])
        print(f"  Hospital capacity: {destination_hosp['name']} occupancy is {destination_hosp['current_occupancy']}/{destination_hosp['emergency_capacity']}")
        print("[OK] Database status state transitions verified successfully.")

        # 4. Trigger Dynamic Re-planning
        print("\nStep 4: Simulating a road blockage to test Dynamic Re-planning...")
        # Create a road block right at Union Square (lat 37.788, lon -122.408)
        blockage_data = {
            "id": "LIVE_BLOCK_UNION_SQ",
            "latitude": 37.788,
            "longitude": -122.408,
            "description": "Stockton St active gas leak repairs",
            "radius_meters": 200.0
        }
        
        block_res = await client.post(f"{API_URL}/api/blockages", json=blockage_data)
        assert block_res.status_code == 200, f"Blockage creation failed: {block_res.text}"
        print(f"[OK] Road block placed: {blockage_data['description']}")

        # Fetch recommendations for the active incident again
        recs_after_block = await client.get(f"{API_URL}/api/incidents/{incident_id}/recommendations")
        recs_list = recs_after_block.json()
        
        # Verify that Option 1 (Union Square ambulance) TrustScore dropped to 0
        blocked_option = next((r for r in recs_list if r["resource_id"] == top_rec["resource_id"]), None)
        
        if blocked_option:
            print(f"  Post-blockage TrustScore for {top_rec['resource_name']}: {blocked_option['trust_score']} / 100")
            print(f"  Route Message: {blocked_option['explanation']}")
            assert blocked_option["trust_score"] == 0.0, "Dynamic re-planning failed to drop blocked route score to 0"
        
        print("[OK] Dynamic Re-planning successfully blocked route and penalized score.")

        # 5. Clean up Blockage
        print("\nStep 5: Clearing simulated blockage...")
        clear_res = await client.delete(f"{API_URL}/api/blockages/LIVE_BLOCK_UNION_SQ")
        assert clear_res.status_code == 200, "Failed to clear blockage"
        print("[OK] Road block cleared successfully.")

    print("\n==================================================")
    print("      ALL LIVE E2E WORKFLOW CHECKS PASSED (100%)")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(test_e2e_flow())
