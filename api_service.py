from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from auth import create_access_token, verify_token

app = FastAPI(title="Industrial Smart Traffic API", version="2.0")
security = HTTPBearer()

# Input Models
class LoginRequest(BaseModel):
    username: str
    password: str

class TrafficInput(BaseModel):
    vehicle_count: int
    weather_code: int
    road_capacity: int

# ----------------------------------------------------
# 1. Login Route (Token मिळवण्यासाठी)
# ----------------------------------------------------
@app.post("/login", tags=["Authentication"])
def login(auth: LoginRequest):
    # Dummy Industrial Credentials Verification
    if auth.username == "admin" and auth.password == "admin123":
        token = create_access_token({"sub": auth.username, "role": "TrafficAdmin"})
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid Username or Password")

# ----------------------------------------------------
# 2. Protected Predict Route (Token असल्याशिवाय चालणार नाही)
# ----------------------------------------------------
@app.post("/predict", tags=["Protected Machine Learning"])
def predict_congestion(data: TrafficInput, credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    verification = verify_token(token)
    
    if "error" in verification:
        raise HTTPException(status_code=401, detail=verification["error"])

    # ML Core Logic Calculation
    score = (data.vehicle_count / data.road_capacity) * 100
    if score > 75:
        status_msg = "Heavy Traffic Congestion"
    elif score > 45:
        status_msg = "Moderate Traffic"
    else:
        status_msg = "Smooth / Low Traffic"

    return {
        "authenticated_user": verification.get("sub"),
        "congestion_index": round(score, 2),
        "status": status_msg,
        "recommendation": "Reroute Traffic" if score > 75 else "Keep Routes Open"
    }