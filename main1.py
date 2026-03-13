from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import re
import os
import joblib
import math
import numpy as np
import pandas as pd
from urllib.parse import urlparse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Load ML Model on Startup
# ─────────────────────────────────────────
MODEL_PATH        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phish_model.pkl")
FEATURES_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_names.pkl")

ml_model          = None
ml_feature_names  = None

if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
    ml_model         = joblib.load(MODEL_PATH)
    ml_feature_names = joblib.load(FEATURES_PATH)
    print("✅ ML model loaded successfully!")
else:
    print("⚠️  ML model not found. Run train_model.py first.")
    print("   Running in heuristics-only mode.\n")


# ─────────────────────────────────────────
# Feature Extraction (same as train_model.py)
# ─────────────────────────────────────────
def extract_features(url):
    try:
        parsed   = urlparse(url)
        hostname = parsed.hostname or ""
        path     = parsed.path or ""
    except:
        hostname = ""
        path     = ""

    features = {}

    features["url_length"]        = len(url)
    features["hostname_length"]   = len(hostname)
    features["path_length"]       = len(path)
    features["dot_count"]         = url.count(".")
    features["hyphen_count"]      = url.count("-")
    features["slash_count"]       = url.count("/")
    features["at_count"]          = url.count("@")
    features["question_count"]    = url.count("?")
    features["eq_count"]          = url.count("=")
    features["underscore_count"]  = url.count("_")
    features["percent_count"]     = url.count("%")
    features["digit_count"]       = sum(c.isdigit() for c in url)
    features["subdomain_count"]   = hostname.count(".")
    features["has_ip"]            = 1 if re.search(r'\d{1,3}(\.\d{1,3}){3}', hostname) else 0
    features["has_at"]            = 1 if "@" in url else 0
    features["has_https"]         = 1 if url.startswith("https") else 0
    features["double_slash"]      = 1 if "//" in path else 0
    features["multi_https"]       = 1 if url.lower().count("https") > 1 else 0
    features["has_redirect"]      = 1 if "redirect" in url.lower() or "url=" in url.lower() else 0
    features["has_encoding"]      = 1 if "%" in url else 0

    keywords = ["login", "verify", "secure", "account", "update",
                "confirm", "banking", "signin", "password", "credential",
                "free", "lucky", "service", "support", "paypal"]
    features["suspicious_keywords"] = sum(kw in url.lower() for kw in keywords)

    suspicious_tlds = [".xyz", ".tk", ".pw", ".gq", ".cf", ".ml", ".ga", ".top", ".click"]
    features["suspicious_tld"]    = 1 if any(hostname.endswith(t) for t in suspicious_tlds) else 0
    features["domain_has_digit"]  = 1 if any(c.isdigit() for c in hostname) else 0
    features["digit_ratio"]       = features["digit_count"] / max(len(url), 1)

    # Advanced features for higher accuracy
    # 1. Shannon Entropy (measures randomness)
    def calculate_entropy(text):
        if not text:
            return 0
        entropy = 0
        for char in set(text):
            p = text.count(char) / len(text)
            entropy -= p * math.log2(p)
        return entropy
    
    features["url_entropy"] = calculate_entropy(url)
    features["hostname_entropy"] = calculate_entropy(hostname)
    
    # 2. Character diversity (unique chars / total chars)
    features["char_diversity"] = len(set(url)) / max(len(url), 1)
    
    # 3. Vowel to consonant ratio
    vowels = sum(1 for c in url.lower() if c in 'aeiou')
    consonants = sum(1 for c in url.lower() if c.isalpha() and c not in 'aeiou')
    features["vowel_ratio"] = vowels / max(vowels + consonants, 1)
    
    # 4. Consecutive character patterns
    features["max_consecutive_digits"] = max([len(s) for s in re.findall(r'\d+', url)] or [0])
    features["max_consecutive_consonants"] = max([len(s) for s in re.findall(r'[bcdfghjklmnpqrstvwxyz]+', url.lower())] or [0])
    
    # 5. Token-based features (split by non-alphanumeric)
    tokens = re.findall(r'\w+', url)
    if tokens:
        features["avg_token_length"] = sum(len(t) for t in tokens) / len(tokens)
        features["max_token_length"] = max(len(t) for t in tokens)
        features["token_count"] = len(tokens)
    else:
        features["avg_token_length"] = 0
        features["max_token_length"] = 0
        features["token_count"] = 0
    
    # 6. Letter case patterns (mixed case can be suspicious)
    features["uppercase_ratio"] = sum(1 for c in url if c.isupper()) / max(len(url), 1)
    features["has_mixed_case"] = 1 if any(c.isupper() for c in hostname) and any(c.islower() for c in hostname) else 0
    
    # 7. Domain-specific features
    domain_parts = hostname.split('.')
    if len(domain_parts) >= 2:
        tld = domain_parts[-1]
        domain = domain_parts[-2]
        features["domain_length"] = len(domain)
        features["tld_length"] = len(tld)
        features["domain_entropy"] = calculate_entropy(domain)
    else:
        features["domain_length"] = 0
        features["tld_length"] = 0
        features["domain_entropy"] = 0
    
    # 8. Anomaly detection features
    features["has_punycode"] = 1 if 'xn--' in hostname else 0
    features["consecutive_dots"] = 1 if '..' in url else 0
    features["starts_with_digit"] = 1 if hostname and hostname[0].isdigit() else 0
    
    # 9. N-gram based features (character patterns)
    # Count suspicious 2-grams and 3-grams
    suspicious_bigrams = ['///', '..', '--', '.-', '-.', '//', '@@']
    features["suspicious_bigram_count"] = sum(url.count(bg) for bg in suspicious_bigrams)
    
    # 10. Advanced TLD analysis
    known_safe_tlds = ['.com', '.org', '.net', '.edu', '.gov', '.mil']
    features["is_safe_tld"] = 1 if any(hostname.endswith(tld) for tld in known_safe_tlds) else 0
    
    # 11. URL depth (number of subdirectories)
    features["url_depth"] = path.count('/')
    
    # 12. Query string analysis
    query = parsed.query or ""
    features["query_length"] = len(query)
    features["param_count"] = query.count('&') + (1 if query else 0)
    
    # 13. Hostname parts analysis
    host_parts = hostname.split('.')
    features["hostname_parts"] = len(host_parts)
    if len(host_parts) > 0:
        features["longest_host_part"] = max(len(part) for part in host_parts)
        features["avg_host_part_length"] = sum(len(part) for part in host_parts) / len(host_parts)
    else:
        features["longest_host_part"] = 0
        features["avg_host_part_length"] = 0
    
    # 14. Statistical features
    # Standard deviation of character positions (measures randomness)
    if url:
        char_positions = [ord(c) for c in url if c.isalnum()]
        if len(char_positions) > 1:
            features["char_pos_std"] = np.std(char_positions)
        else:
            features["char_pos_std"] = 0
    else:
        features["char_pos_std"] = 0
    
    # 15. Phishing pattern score (combination of multiple signals)
    phishing_score = 0
    if features["has_ip"]: phishing_score += 3
    if features["has_at"]: phishing_score += 2
    if features["suspicious_tld"]: phishing_score += 2
    if features["url_length"] > 75: phishing_score += 2
    if features["subdomain_count"] > 3: phishing_score += 2
    if features["has_punycode"]: phishing_score += 3
    if features["domain_has_digit"]: phishing_score += 1
    features["phishing_pattern_score"] = phishing_score
    
    # 16. Lexical diversity (unique words / total words)
    url_tokens = re.findall(r'\w+', url.lower())
    if url_tokens:
        features["lexical_diversity"] = len(set(url_tokens)) / len(url_tokens)
    else:
        features["lexical_diversity"] = 0
    
    # 17. Brand name detection (common phishing targets)
    brand_keywords = ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'facebook', 
                      'netflix', 'instagram', 'twitter', 'linkedin', 'ebay', 'banking']
    features["has_brand_name"] = 1 if any(brand in url.lower() for brand in brand_keywords) else 0
    features["brand_not_in_domain"] = 1 if (features["has_brand_name"] and not any(brand in hostname.lower() for brand in brand_keywords)) else 0

    # Engineered interaction features
    features["length_digit_ratio"] = features["url_length"] * features["digit_ratio"]
    features["suspicious_score"] = features["suspicious_keywords"] + features["suspicious_tld"] + features["has_ip"] + features["has_at"]
    features["complexity_score"] = features["dot_count"] + features["slash_count"] + features["hyphen_count"] + features["underscore_count"]
    features["has_long_suspicious"] = 1 if (features["url_length"] > 50 and features["suspicious_keywords"] > 0) else 0
    features["subdomain_dot_ratio"] = features["subdomain_count"] / max(features["dot_count"], 1)
    features["special_char_count"] = features["at_count"] + features["question_count"] + features["eq_count"] + features["percent_count"]
    features["path_to_url_ratio"] = features["path_length"] / max(features["url_length"], 1)
    
    # Advanced interaction features
    features["entropy_diversity"] = features["url_entropy"] * features["char_diversity"]
    features["hostname_domain_ratio"] = features["hostname_length"] / (features["domain_length"] + 1)
    features["suspicious_entropy"] = features["suspicious_score"] * features["hostname_entropy"]
    features["token_complexity"] = features["token_count"] * features["complexity_score"]
    
    # New sophisticated interactions
    features["phishing_entropy_score"] = features["phishing_pattern_score"] * features["url_entropy"]
    features["brand_suspicious_score"] = features["brand_not_in_domain"] * features["suspicious_score"]
    features["depth_complexity"] = features["url_depth"] * features["complexity_score"]
    features["query_suspicious"] = features["query_length"] * features["suspicious_keywords"]
    features["lexical_entropy"] = features["lexical_diversity"] * features["hostname_entropy"]
    features["risk_aggregation"] = (features["phishing_pattern_score"] + features["suspicious_score"] + 
                                     features["suspicious_bigram_count"] + features["brand_not_in_domain"])

    return features


# ─────────────────────────────────────────
# Heuristic Checks
# ─────────────────────────────────────────
def run_heuristics(url, hostname):
    risk_points = 0
    flags = []

    if len(url) > 75:
        risk_points += 20
        flags.append("Excessive Length")

    if "@" in url:
        risk_points += 40
        flags.append("Contains @ Symbol")

    if re.search(r'\d{1,3}(\.\d{1,3}){3}', hostname):
        risk_points += 50
        flags.append("Uses IP Address")

    if hostname.count('.') > 2:
        risk_points += 20
        flags.append("High Subdomain Count")

    if url.lower().count('https') > 1:
        risk_points += 25
        flags.append("Multiple HTTPS Tokens")

    suspicious_keywords = ["login", "verify", "secure", "account", "update",
                           "confirm", "banking", "signin", "password", "credential"]
    if any(kw in url.lower() for kw in suspicious_keywords):
        risk_points += 20
        flags.append("Suspicious Keywords")

    suspicious_tlds = [".xyz", ".tk", ".pw", ".gq", ".cf", ".ml", ".ga", ".top", ".click"]
    if any(hostname.endswith(tld) for tld in suspicious_tlds):
        risk_points += 30
        flags.append("Suspicious TLD")

    if hostname.count('-') > 2:
        risk_points += 15
        flags.append("Excessive Hyphens in Domain")

    if "redirect" in url.lower() or "url=" in url.lower():
        risk_points += 25
        flags.append("Open Redirect Pattern")

    if url.count('%') > 3:
        risk_points += 20
        flags.append("Excessive URL Encoding")

    typosquat_patterns = [
        r'paypa[^l]', r'g(?:o{1}|o{3,})gle', r'face(?:b[o0]{2,}|bo{2,})k', r'micr[o0]s[o0]ft',
        r'app[l1]e', r'amaz[o0]n', r'netfl[i1]x', r'ins[t7]agram'
    ]
    if any(re.search(p, hostname.lower()) for p in typosquat_patterns):
        risk_points += 40
        flags.append("Possible Typosquatting")

    return min(risk_points, 100), flags


# ─────────────────────────────────────────
# Status from Score
# ─────────────────────────────────────────
def get_status(score):
    if score >= 70:
        return "Malicious"
    elif score >= 40:
        return "Suspicious"
    return "Safe"


# ─────────────────────────────────────────
# Request Model
# ─────────────────────────────────────────
class URLRequest(BaseModel):
    url: str


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.get("/")
async def serve_index():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    return FileResponse(html_path)


@app.post("/analyze")
async def analyze(request: URLRequest):
    url = request.url.strip()

    # Validate URL
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid URL. Please include http:// or https://"}
        )

    hostname = parsed.hostname or ""
    
    # Whitelist of known legitimate domains
    trusted_domains = [
        'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
        'linkedin.com', 'microsoft.com', 'apple.com', 'amazon.com', 'netflix.com',
        'github.com', 'stackoverflow.com', 'reddit.com', 'wikipedia.org', 'bbc.com',
        'cnn.com', 'nytimes.com', 'amazon.in', 'flipkart.com', 'ebay.com'
    ]
    
    # Check if domain is whitelisted
    is_whitelisted = any(hostname.endswith(domain) or hostname == domain for domain in trusted_domains)
    
    # If whitelisted, return safe immediately
    if is_whitelisted:
        return {
            "url":              url,
            "risk_score":       0,
            "status":           "Safe",
            "features":         ["Verified Trusted Domain"],
            "heuristic_score":  0,
            "ml_score":         0.0,
            "ml_verdict":       "Safe",
            "ml_confidence":    100.0,
            "ml_available":     ml_model is not None,
        }

    # ── Heuristic Score ──
    heuristic_score, flags = run_heuristics(url, hostname)

    # ── ML Score ──
    ml_score      = None
    ml_confidence = None
    ml_verdict    = None

    if ml_model is not None:
        try:
            feats      = extract_features(url)
            fdf        = pd.DataFrame([feats])[ml_feature_names]
            prob       = ml_model.predict_proba(fdf)[0][1]   # probability of phishing
            ml_score   = round(prob * 100, 1)
            ml_verdict = "Phishing" if prob > 0.5 else "Safe"
            ml_confidence = round(max(prob, 1 - prob) * 100, 1)
        except Exception as e:
            ml_score = None
            print(f"ML prediction error: {e}")

    # ── Combined Final Score ──
    if ml_score is not None:
        # 60% ML weight + 40% heuristic weight
        final_score = round((ml_score * 0.6) + (heuristic_score * 0.4))
    else:
        final_score = heuristic_score

    final_score = min(final_score, 100)
    status      = get_status(final_score)

    return {
        "url":              url,
        "risk_score":       final_score,
        "status":           status,
        "features":         flags,
        "heuristic_score":  heuristic_score,
        "ml_score":         ml_score,
        "ml_verdict":       ml_verdict,
        "ml_confidence":    ml_confidence,
        "ml_available":     ml_model is not None,
    }


if __name__ == "__main__":
    import uvicorn
    print("\n✅ PhishGuard is running!")
    print("👉 Open in browser: http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
