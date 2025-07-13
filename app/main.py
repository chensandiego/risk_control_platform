from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pymongo
import redis
from typing import Optional, List
from datetime import datetime
import re
import math
from urllib.parse import urlparse
import requests
import socket
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# --- Mount Static Files ---
app.mount("/static", StaticFiles(directory="static"), name="static")



# --- Constants ---
RATE_LIMIT_COUNT = 10  # 10 requests
RATE_LIMIT_WINDOW_SECONDS = 60  # per 60 seconds
ABUSEIPDB_API_KEY = os.getenv("ABUSEIPDB_API_KEY", "YOUR_ABUSEIPDB_API_KEY")


# --- Data Models ---
class URLCheckRequest(BaseModel):
    url: str

class URLCheckResponse(BaseModel):
    url: str
    risk: str
    reason: Optional[str] = None

class BlocklistRequest(BaseModel):
    url: str
    category: str

class TrafficEvent(BaseModel):
    ip: str
    timestamp: datetime
    event: str

class LabeledURLRequest(BaseModel):
    url: str
    label: str # e.g., 'phishing', 'legitimate', 'malware'

class KeywordAddRequest(BaseModel):
    keyword: str

class KeywordRemoveRequest(BaseModel):
    keyword: str

# --- Helper Functions ---


def run_heuristic_checks(url: str, db) -> Optional[str]:
    """Performs a series of heuristic checks on a URL."""
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname

    # Fetch keywords directly from MongoDB
    keywords_collection = db.suspicious_keywords_config
    current_keywords = [doc["keyword"] for doc in keywords_collection.find()]

    for keyword in current_keywords:
        if keyword in url.lower():
            return f"heuristic_suspicious_keyword_{keyword}"

    if hostname and re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname):
        return "heuristic_hostname_is_ip"

    if hostname and len(hostname.split('.')) > 4:
        return "heuristic_too_many_subdomains"

    if hostname and is_dga_domain(hostname):
        return "heuristic_dga_detected"

    return None


def check_abuseipdb(ip: str) -> Optional[str]:
    """Checks an IP address against the AbuseIPDB API."""
    if not ABUSEIPDB_API_KEY or ABUSEIPDB_API_KEY == "YOUR_ABUSEIPDB_API_KEY":
        return None  # Skip if API key is not configured

    try:
        response = requests.get(
            "https://api.abuseipdb.com/api/v2/check",
            params={"ipAddress": ip, "maxAgeInDays": "90"},
            headers={"Key": ABUSEIPDB_API_KEY, "Accept": "application/json"},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json().get("data", {})

        if data.get("abuseConfidenceScore", 0) >= 90:
            return f"abuseipdb_high_confidence_{data['abuseConfidenceScore']}"
            
    except requests.RequestException as e:
        print(f"Error querying AbuseIPDB: {e}")
        return None
        
    return None

def calculate_entropy(s: str) -> float:
    """Calculate the Shannon entropy of a string."""
    if not s:
        return 0.0
    
    # Count character frequencies
    char_counts = {}
    for char in s:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Calculate entropy
    entropy = 0.0
    total_chars = len(s)
    for count in char_counts.values():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
    
    return entropy

def is_dga_domain(domain: str) -> bool:
    """
    Simple heuristic-based DGA detection.
    Checks for high entropy and lack of common dictionary words.
    """
    # Exclude common TLDs and known legitimate patterns
    if "." not in domain:
        return False # Not a valid domain format for this check

    parts = domain.split('.')
    # Consider only the domain name, not TLD
    domain_name = parts[0] 
    
    # Heuristic 1: Check for high entropy
    # DGA domains often have high entropy due to random character generation
    entropy = calculate_entropy(domain_name)
    if entropy > 3.5:  # Threshold can be tuned
        return True

    # Heuristic 2: Check for unusual length (too short or too long)
    if len(domain_name) < 6 or len(domain_name) > 20: # Example thresholds
        return True

    # Heuristic 3: Check for presence of too many consecutive consonants or vowels
    # This can indicate non-pronounceable, random-like strings
    vowels = "aeiou"
    consonants = "bcdfghjklmnpqrstvwxyz"
    
    max_consecutive_vowels = 0
    current_consecutive_vowels = 0
    max_consecutive_consonants = 0
    current_consecutive_consonants = 0

    for char in domain_name.lower():
        if char in vowels:
            current_consecutive_vowels += 1
            current_consecutive_consonants = 0
        elif char in consonants:
            current_consecutive_consonants += 1
            current_consecutive_vowels = 0
        else: # Reset for non-alphabetic characters
            current_consecutive_vowels = 0
            current_consecutive_consonants = 0
        
        max_consecutive_vowels = max(max_consecutive_vowels, current_consecutive_vowels)
        max_consecutive_consonants = max(max_consecutive_consonants, current_consecutive_consonants)

    if max_consecutive_vowels > 4 or max_consecutive_consonants > 5: # Example thresholds
        return True

    return False

def extract_url_features(url: str) -> dict:
    """Extracts basic features from a URL for ML model training."""
    return {
        "url_length": len(url),
        "num_special_chars": len(re.findall(r'[^a-zA-Z0-9]', url)),
        "num_digits": len(re.findall(r'\d', url)),
    }

# --- Database Connection & Startup ---
@app.on_event("startup")
def startup_db_client():
    app.mongodb_client = pymongo.MongoClient("mongodb://mongodb:27017/")
    app.redis_client = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)
    
    print("Connected to the MongoDB and Redis databases!")

@app.on_event("shutdown")
def shutdown_db_client():
    app.mongodb_client.close()
    print("MongoDB connection closed.")

# --- Middleware for Traffic Anti-Fraud ---
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/static"):
        return await call_next(request)
        
    client_ip = request.client.host
    
    pipe = app.redis_client.pipeline()
    pipe.incr(client_ip)
    pipe.expire(client_ip, RATE_LIMIT_WINDOW_SECONDS)
    request_count = pipe.execute()[0]

    if request_count > RATE_LIMIT_COUNT:
        db = app.mongodb_client.risk_control
        traffic_events_collection = db.traffic_events
        traffic_events_collection.insert_one({
            "ip": client_ip,
            "timestamp": datetime.utcnow(),
            "event": "rate_limit_exceeded"
        })
        return Response(content="Too Many Requests", status_code=429)
    
    response = await call_next(request)
    return response

# --- API Endpoints ---
@app.get("/", response_class=FileResponse)
async def read_index():
    return 'static/index.html'

@app.post("/api/url/check", response_model=URLCheckResponse)
def check_url(request: URLCheckRequest):
    url_to_check = request.url
    
    cached_result = app.redis_client.get(f"url:{url_to_check}")
    if cached_result:
        risk, reason = cached_result.split(":", 1)
        return URLCheckResponse(url=url_to_check, risk=risk, reason=reason)

    db = app.mongodb_client.risk_control
    blocklist_collection = db.url_blocklist
    found_item = blocklist_collection.find_one({"url": url_to_check})
    
    if found_item:
        risk = "high"
        reason = f"blocklisted_{found_item['category']}"
    else:
        # Resolve hostname to IP for AbuseIPDB check
        try:
            hostname = urlparse(url_to_check).hostname
            ip_address = socket.gethostbyname(hostname)
            abuse_reason = check_abuseipdb(ip_address)
            if abuse_reason:
                risk = "high"
                reason = abuse_reason
            else:
                heuristic_reason = run_heuristic_checks(url_to_check, db)
                if heuristic_reason:
                    risk = "medium"
                    reason = heuristic_reason
                else:
                    risk = "low"
                    reason = "not_blocklisted"
        except (socket.gaierror, TypeError):
            heuristic_reason = run_heuristic_checks(url_to_check, db)
            if heuristic_reason:
                risk = "medium"
                reason = heuristic_reason
            else:
                risk = "low"
                reason = "not_blocklisted"

    app.redis_client.set(f"url:{url_to_check}", f"{risk}:{reason}", ex=3600)
    return URLCheckResponse(url=url_to_check, risk=risk, reason=reason)

@app.post("/api/url/blocklist")
def add_to_blocklist(request: BlocklistRequest):
    db = app.mongodb_client.risk_control
    blocklist_collection = db.url_blocklist
    blocklist_collection.update_one(
        {"url": request.url},
        {"$set": {"category": request.category}},
        upsert=True
    )
    app.redis_client.delete(f"url:{request.url}")
    return {"message": f"URL '{request.url}' has been added/updated in the blocklist."}

@app.post("/api/url/log_and_label")
def log_and_label_url(request: LabeledURLRequest):
    """Logs URL features and a label for ML training."""
    db = app.mongodb_client.risk_control
    training_data_collection = db.url_training_data
    
    features = extract_url_features(request.url)
    features["label"] = request.label
    features["url"] = request.url
    
    training_data_collection.insert_one(features)
    return {"message": "URL logged for training."}

@app.post("/api/keywords/add")
def add_keyword(request: KeywordAddRequest):
    db = app.mongodb_client.risk_control
    keywords_collection = db.suspicious_keywords_config
    keywords_collection.update_one(
        {"keyword": request.keyword},
        {"$set": {"keyword": request.keyword}},
        upsert=True
    )
    # Keywords will be reloaded on next service restart
    return {"message": f"Keyword '{request.keyword}' added."}

@app.delete("/api/keywords/remove")
def remove_keyword(request: KeywordRemoveRequest):
    db = app.mongodb_client.risk_control
    keywords_collection = db.suspicious_keywords_config
    result = keywords_collection.delete_one({"keyword": request.keyword})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Keyword not found.")
    # Keywords will be reloaded on next service restart
    return {"message": f"Keyword '{request.keyword}' removed."}

@app.get("/api/keywords/list")
def list_keywords():
    db = app.mongodb_client.risk_control
    keywords_collection = db.suspicious_keywords_config
    current_keywords = [doc["keyword"] for doc in keywords_collection.find()]
    return {"keywords": current_keywords}

@app.get("/api/events", response_model=List[TrafficEvent])
def get_traffic_events():
    db = app.mongodb_client.risk_control
    traffic_events_collection = db.traffic_events
    events = traffic_events_collection.find().sort("timestamp", -1).limit(10)
    return list(events)