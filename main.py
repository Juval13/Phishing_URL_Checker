from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import re
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str

@app.get("/")
async def serve_index():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    return FileResponse(html_path)

@app.post("/analyze")
async def analyze(request: URLRequest):
    url = request.url
    risk_points = 0
    features = []

    if len(url) > 75:
        risk_points += 30
        features.append("Excessive Length")

    if "@" in url:
        risk_points += 40
        features.append("Contains @ symbol")

    if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
        risk_points += 50
        features.append("Uses IP Address")

    if url.count('.') > 3:
        risk_points += 20
        features.append("High Subdomain Count")

    if url.count('https') > 1:
        risk_points += 25
        features.append("Multiple HTTPS tokens")

    final_score = min(risk_points, 100)
    status = "Malicious" if final_score > 40 else "Safe"

    return {
        "url": url,
        "risk_score": final_score,
        "status": status,
        "features": features
    }

if __name__ == "__main__":
    import uvicorn
    print("\n✅ PhishGuard is running!")
    print("👉 Open in browser: http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)