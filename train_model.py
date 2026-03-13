import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import math
from urllib.parse import urlparse
import urllib.request
import os
from collections import Counter

# ─────────────────────────────────────────
# STEP 1: Use Enhanced Phishing Dataset
# ─────────────────────────────────────────
# Using comprehensive enhanced dataset with balanced good/bad URLs
DATASET_FILE = "phishing_dataset_enhanced.csv"

print("📥 Loading enhanced phishing dataset...")
if not os.path.exists(DATASET_FILE):
    print(f"❌ Dataset file '{DATASET_FILE}' not found!")
    print("   Please ensure phishing_dataset_enhanced.csv is in the current directory.")
    exit()
print("✅ Enhanced dataset loaded successfully!\n")

# ─────────────────────────────────────────
# STEP 2: Feature Extraction from URL
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# STEP 3: Load and Prepare Dataset
# ─────────────────────────────────────────
print("📊 Loading dataset...")
df = pd.read_csv(DATASET_FILE)
print(f"   Columns found: {list(df.columns)}")
print(f"   Total rows: {len(df)}\n")

# Auto-detect URL column and label column
url_col   = None
label_col = None

for col in df.columns:
    if col.lower() in ["url", "urls", "link", "address"]:
        url_col = col
    if col.lower() in ["label", "status", "type", "class", "phishing", "result"]:
        label_col = col

if url_col is None or label_col is None:
    print(f"❌ Could not auto-detect columns. Found: {list(df.columns)}")
    print("   Please set url_col and label_col manually in the script.")
    exit()

print(f"✅ Using URL column: '{url_col}' | Label column: '{label_col}'")
print(f"   Label distribution:\n{df[label_col].value_counts()}\n")

# Drop rows with missing values
df = df[[url_col, label_col]].dropna()

# Normalize labels to 0 (safe) and 1 (phishing)
unique_labels = df[label_col].unique()
print(f"   Unique label values: {unique_labels}")

# Handle both numeric and string labels
label_map = {}
for lbl in unique_labels:
    lbl_str = str(lbl).lower().strip()
    if lbl_str in ["1", "phishing", "bad", "malicious", "phishy", "yes"]:
        label_map[lbl] = 1
    elif lbl_str in ["0", "legitimate", "good", "safe", "benign", "no"]:
        label_map[lbl] = 0
    else:
        label_map[lbl] = 0

df["label_encoded"] = df[label_col].map(label_map)
print(f"   Encoded distribution:\n{df['label_encoded'].value_counts()}\n")

# Use 100% of dataset for maximum accuracy
print(f"🔺 Using {len(df):,} samples (100% of enhanced dataset) for maximum accuracy\n")

# ─────────────────────────────────────────
# STEP 4: Extract Features
# ─────────────────────────────────────────
print("⚙️  Extracting features from URLs (this may take a moment)...")
feature_list = df[url_col].apply(extract_features).tolist()
X = pd.DataFrame(feature_list)
y = df["label_encoded"].values

print(f"✅ Features extracted: {X.shape[1]} features from {X.shape[0]} URLs\n")
print(f"   Feature names: {list(X.columns)}\n")

# ─────────────────────────────────────────
# STEP 4.5: Add Engineered Features for Higher Accuracy
# ─────────────────────────────────────────
print("⚙️  Adding advanced engineered features for improved accuracy...")

# Interaction features
X['length_digit_ratio'] = X['url_length'] * X['digit_ratio']
X['suspicious_score'] = X['suspicious_keywords'] + X['suspicious_tld'] + X['has_ip'] + X['has_at']
X['complexity_score'] = X['dot_count'] + X['slash_count'] + X['hyphen_count'] + X['underscore_count']
X['has_long_suspicious'] = ((X['url_length'] > 50) & (X['suspicious_keywords'] > 0)).astype(int)
X['subdomain_dot_ratio'] = X['subdomain_count'] / (X['dot_count'] + 1)
X['special_char_count'] = X['at_count'] + X['question_count'] + X['eq_count'] + X['percent_count']
X['path_to_url_ratio'] = X['path_length'] / (X['url_length'] + 1)

# Advanced interaction features
X['entropy_diversity'] = X['url_entropy'] * X['char_diversity']
X['hostname_domain_ratio'] = X['hostname_length'] / (X['domain_length'] + 1)
X['suspicious_entropy'] = X['suspicious_score'] * X['hostname_entropy']
X['token_complexity'] = X['token_count'] * X['complexity_score']

# New sophisticated interactions
X['phishing_entropy_score'] = X['phishing_pattern_score'] * X['url_entropy']
X['brand_suspicious_score'] = X['brand_not_in_domain'] * X['suspicious_score']
X['depth_complexity'] = X['url_depth'] * X['complexity_score']
X['query_suspicious'] = X['query_length'] * X['suspicious_keywords']
X['lexical_entropy'] = X['lexical_diversity'] * X['hostname_entropy']
X['risk_aggregation'] = (X['phishing_pattern_score'] + X['suspicious_score'] + 
                         X['suspicious_bigram_count'] + X['brand_not_in_domain'])

print(f"✅ Added {X.shape[1] - 41} engineered features. Total: {X.shape[1]} features\n")

# ─────────────────────────────────────────
# STEP 5: Train / Test Split
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"📦 Train size: {len(X_train)} | Test size: {len(X_test)}\n")

# ─────────────────────────────────────────
# STEP 6: Train Optimized Random Forest Model
# ─────────────────────────────────────────
print("🌲 Training optimized Random Forest model with 71 advanced features...")

# Random Forest - optimized with extensive features for enhanced dataset
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=28,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    class_weight='balanced',
    max_samples=0.90,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)
print("✅ Model trained!\n")
print(f"🌲 Out-of-bag score: {model.oob_score_*100:.2f}%\n")

# ─────────────────────────────────────────
# STEP 7: Evaluate Model
# ─────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print(f"📈 MODEL ACCURACY: {accuracy * 100:.2f}%")
print("=" * 50)
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

# Feature importance
print("🔍 Top 10 Most Important Features:")
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
print(importances.head(10).to_string())
print()

# ─────────────────────────────────────────
# STEP 8: Save Model and Feature Names
# ─────────────────────────────────────────
joblib.dump(model, "phish_model.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print("\n✅ Model saved as: phish_model.pkl")
print("✅ Feature names saved as: feature_names.pkl")
print("\n🎉 Training complete! You can now use the model in main.py")

# ─────────────────────────────────────────
# STEP 9: Quick Sanity Test
# ─────────────────────────────────────────
print("\n🧪 Quick sanity test:")
test_urls = [
    "https://google.com",
    "https://github.com",
    "http://192.168.1.1/login@phish.xyz",
    "http://paypal-secure-login.verify.account.xyz/confirm",
    "https://amazon.com",
]

loaded_model    = joblib.load("phish_model.pkl")
loaded_features = joblib.load("feature_names.pkl")

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
    
    return fdf[loaded_features]

for test_url in test_urls:
    fdf = prepare_features(test_url)
    prob = loaded_model.predict_proba(fdf)[0][1]
    result = "🔴 Phishing" if prob > 0.5 else "🟢 Safe"
    print(f"   {result} ({prob*100:.1f}% risk) — {test_url}")
