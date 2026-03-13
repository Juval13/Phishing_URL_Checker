from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import re
import os
import joblib
import pandas as pd
import numpy as np
import math
from urllib.parse import urlparse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained ML model
try:
    model = joblib.load("phish_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    print("✅ ML Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None
    feature_names = None

class URLRequest(BaseModel):
    url: str

def extract_features(url):
    """Extract numerical features from a URL string."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
    except:
        hostname = ""
        path = ""

    features = {}

    # Length-based features
    features["url_length"]      = len(url)
    features["hostname_length"] = len(hostname)
    features["path_length"]     = len(path)

    # Count-based features
    features["dot_count"]       = url.count(".")
    features["hyphen_count"]    = url.count("-")
    features["slash_count"]     = url.count("/")
    features["at_count"]        = url.count("@")
    features["question_count"]  = url.count("?")
    features["eq_count"]        = url.count("=")
    features["underscore_count"]= url.count("_")
    features["percent_count"]   = url.count("%")
    features["digit_count"]     = sum(c.isdigit() for c in url)
    features["subdomain_count"] = hostname.count(".")

    # Boolean features (0 or 1)
    features["has_ip"]          = 1 if re.search(r'\d{1,3}(\.\d{1,3}){3}', hostname) else 0
    features["has_at"]          = 1 if "@" in url else 0
    features["has_https"]       = 1 if url.startswith("https") else 0
    features["double_slash"]    = 1 if "//" in path else 0
    features["multi_https"]     = 1 if url.lower().count("https") > 1 else 0
    features["has_redirect"]    = 1 if "redirect" in url.lower() or "url=" in url.lower() else 0
    features["has_encoding"]    = 1 if "%" in url else 0

    # Suspicious keyword features
    keywords = ["login", "verify", "secure", "account", "update",
                "confirm", "banking", "signin", "password", "credential",
                "free", "lucky", "service", "support", "paypal"]
    features["suspicious_keywords"] = sum(kw in url.lower() for kw in keywords)

    # Suspicious TLD
    suspicious_tlds = [".xyz", ".tk", ".pw", ".gq", ".cf", ".ml", ".ga", ".top", ".click"]
    features["suspicious_tld"]  = 1 if any(hostname.endswith(t) for t in suspicious_tlds) else 0

    # Domain has digits
    features["domain_has_digit"] = 1 if any(c.isdigit() for c in hostname) else 0

    # Ratio of digits to URL length
    features["digit_ratio"] = features["digit_count"] / max(len(url), 1)

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

    return features

def prepare_features(url):
    """Extract features and add engineered features"""
    feats = extract_features(url)
    fdf = pd.DataFrame([feats])
    
    # Add engineered features (same as training)
    fdf['length_digit_ratio'] = fdf['url_length'] * fdf['digit_ratio']
    fdf['suspicious_score'] = fdf['suspicious_keywords'] + fdf['suspicious_tld'] + fdf['has_ip'] + fdf['has_at']
    fdf['complexity_score'] = fdf['dot_count'] + fdf['slash_count'] + fdf['hyphen_count'] + fdf['underscore_count']
    fdf['has_long_suspicious'] = ((fdf['url_length'] > 50) & (fdf['suspicious_keywords'] > 0)).astype(int)
    fdf['subdomain_dot_ratio'] = fdf['subdomain_count'] / (fdf['dot_count'] + 1)
    fdf['special_char_count'] = fdf['at_count'] + fdf['question_count'] + fdf['eq_count'] + fdf['percent_count']
    fdf['path_to_url_ratio'] = fdf['path_length'] / (fdf['url_length'] + 1)
    fdf['entropy_diversity'] = fdf['url_entropy'] * fdf['char_diversity']
    fdf['hostname_domain_ratio'] = fdf['hostname_length'] / (fdf['domain_length'] + 1)
    fdf['suspicious_entropy'] = fdf['suspicious_score'] * fdf['hostname_entropy']
    fdf['token_complexity'] = fdf['token_count'] * fdf['complexity_score']
    fdf['phishing_entropy_score'] = fdf['phishing_pattern_score'] * fdf['url_entropy']
    fdf['brand_suspicious_score'] = fdf['brand_not_in_domain'] * fdf['suspicious_score']
    fdf['depth_complexity'] = fdf['url_depth'] * fdf['complexity_score']
    fdf['query_suspicious'] = fdf['query_length'] * fdf['suspicious_keywords']
    fdf['lexical_entropy'] = fdf['lexical_diversity'] * fdf['hostname_entropy']
    fdf['risk_aggregation'] = (fdf['phishing_pattern_score'] + fdf['suspicious_score'] + 
                               fdf['suspicious_bigram_count'] + fdf['brand_not_in_domain'])
    
    return fdf[feature_names]

@app.get("/")
async def serve_index():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    return FileResponse(html_path)

@app.post("/analyze")
async def analyze(request: URLRequest):
    url = request.url
    
    if model is None:
        return {
            "url": url,
            "risk_score": 0,
            "status": "Error: Model not loaded",
            "features": []
        }
    
    try:
        # List of well-known safe domains
        known_safe_domains = [
            'google.com', 'youtube.com', 'facebook.com', 'amazon.com', 'wikipedia.org',
            'twitter.com', 'instagram.com', 'linkedin.com', 'reddit.com', 'netflix.com',
            'microsoft.com', 'apple.com', 'github.com', 'stackoverflow.com', 'yahoo.com',
            'zoom.us', 'whatsapp.com', 'tiktok.com', 'discord.com', 'twitch.tv',
            'ebay.com', 'paypal.com', 'dropbox.com', 'adobe.com', 'salesforce.com',
            'oracle.com', 'ibm.com', 'cisco.com', 'samsung.com', 'sony.com',
            'walmart.com', 'target.com', 'bestbuy.com', 'airbnb.com', 'booking.com',
            'spotify.com', 'soundcloud.com', 'pinterest.com', 'tumblr.com', 'flickr.com'
        ]
        
        # Check if URL is from a known safe domain
        parsed = urlparse(url)
        hostname = (parsed.hostname or '').lower()
        is_known_safe = any(hostname.endswith(domain) or hostname == domain for domain in known_safe_domains)
        
        if is_known_safe:
            return {
                "url": url,
                "risk_score": 0,
                "status": "Safe",
                "features": ["Trusted website"]
            }
        
        # Extract features and predict using ML model
        fdf = prepare_features(url)
        probability = model.predict_proba(fdf)[0][1]  # Probability of phishing
        risk_score = int(probability * 100)
        status = "Malicious" if probability > 0.5 else "Safe"
        
        # Get top suspicious features for explanation
        feature_vals = fdf.iloc[0].to_dict()
        suspicious_features = []
        
        if feature_vals.get('has_ip', 0) == 1:
            suspicious_features.append("Uses IP Address")
        if feature_vals.get('has_at', 0) == 1:
            suspicious_features.append("Contains @ symbol")
        if feature_vals.get('suspicious_tld', 0) == 1:
            suspicious_features.append("Suspicious TLD")
        if feature_vals.get('url_length', 0) > 75:
            suspicious_features.append("Excessive Length")
        if feature_vals.get('suspicious_keywords', 0) > 0:
            suspicious_features.append(f"Suspicious Keywords ({int(feature_vals['suspicious_keywords'])})")
        if feature_vals.get('brand_not_in_domain', 0) == 1:
            suspicious_features.append("Brand Name Spoofing")
        if feature_vals.get('has_https', 0) == 0:
            suspicious_features.append("No HTTPS")
        if feature_vals.get('subdomain_count', 0) > 3:
            suspicious_features.append("High Subdomain Count")
        
        return {
            "url": url,
            "risk_score": risk_score,
            "status": status,
            "features": suspicious_features if suspicious_features else ["No major red flags detected"]
        }
    except Exception as e:
        return {
            "url": url,
            "risk_score": 0,
            "status": f"Error: {str(e)}",
            "features": []
        }

if __name__ == "__main__":
    import uvicorn
    print("\n✅ PhishGuard is running!")
    print("👉 Open in browser: http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)