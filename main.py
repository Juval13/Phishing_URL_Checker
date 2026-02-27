from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import re

app = FastAPI()

# Enable CORS so the HTML file can talk to the Python server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str

@app.post("/analyze")
async def analyze(request: URLRequest):
    url = request.url
    risk_points = 0
    features = []

    # Rule 1: Long URLs are suspicious
    if len(url) > 75:
        risk_points += 30
        features.append("Excessive Length")

    # Rule 2: Presence of '@' symbol
    if "@" in url:
        risk_points += 40
        features.append("Contains @ symbol")

    # Rule 3: Presence of IP address instead of domain
    if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
        risk_points += 50
        features.append("Uses IP Address")

    # Rule 4: Too many dots (subdomain abuse)
    if url.count('.') > 3:
        risk_points += 20
        features.append("High Subdomain Count")

    # Rule 5: Presence of "HTTPS" (Only suspicious if it's in the wrong place)
    if url.count('https') > 1:
        risk_points += 25
        features.append("Multiple HTTPS tokens")

    # Final calculation
    final_score = min(risk_points, 100) # Cap at 100%
    status = "Malicious" if final_score > 40 else "Safe"

    return {
        "url": url,
        "risk_score": final_score,
        "status": status,
        "features": features
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)