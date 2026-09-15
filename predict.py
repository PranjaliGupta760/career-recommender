import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack

# STEP 1 — Load all saved models and encoders

with open('models/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    skills_vectorizer = pickle.load(f)

with open('models/title_vectorizer.pkl', 'rb') as f:
    title_vectorizer = pickle.load(f)

with open('models/desc_vectorizer.pkl', 'rb') as f:
    desc_vectorizer = pickle.load(f)

with open('models/action_vectorizer.pkl', 'rb') as f:
    action_vectorizer = pickle.load(f)

print(" All models loaded successfully!")
print(f"   Industries: {list(encoders['industry'].classes_)}")

# STEP 2 — Take Real User Input

print("\n" + "="*50)
print("   JOB INDUSTRY PREDICTOR")
print("="*50)

job_title    = input("\nEnter Job Title:        ")
skills       = input("Enter Skills (comma separated): ")
description  = input("Enter Job Description:  ")
experience   = input("Enter Experience Level  (Entry level / Mid-Senior level / Associate / Director / Internship / Executive): ")
location     = input("Enter Location (e.g. New York, NY): ")

# STEP 3 — Preprocess Input

# Clean location same way as training
location_state = location.split(',')[-1].strip()

# Encode experience level
# Handle unseen experience levels gracefully
if experience in encoders['experience'].classes_:
    exp_encoded = encoders['experience'].transform([experience])[0]
else:
    print(f"\n Unknown experience level '{experience}' — defaulting to Entry level")
    exp_encoded = encoders['experience'].transform(['Entry level'])[0]

# Encode location
# Handle unseen locations gracefully
if location_state in encoders['location'].classes_:
    loc_encoded = encoders['location'].transform([location_state])[0]
else:
    print(f"\n Unknown location '{location_state}' — defaulting to most common")
    loc_encoded = 0  # default

# STEP 4 — Vectorize Text Features
def extract_action_phrases(text):
    import re
    ACTION_VERBS = re.compile(
        r'\b(design|build|develop|manage|lead|analyze|create|implement|'
        r'source|screen|recruit|hire|interview|partner|coordinate|'
        r'administer|provide|monitor|assess|conduct|maintain|optimize|'
        r'collaborate|integrate|deploy|architect|deliver|support|write|'
        r'research|plan|execute|review|own|drive|define|establish)\b',
        re.IGNORECASE
    )
    sentences = re.split(r'[.!?\n]', text)
    action_sentences = [s.strip() for s in sentences if ACTION_VERBS.search(s.strip())]
    return ' '.join(action_sentences) if action_sentences else text

action_phrases  = extract_action_phrases(description)
action_tfidf    = action_vectorizer.transform([action_phrases])
skills_tfidf    = skills_vectorizer.transform([skills])
title_tfidf     = title_vectorizer.transform([job_title])
desc_tfidf      = desc_vectorizer.transform([description])

numeric_features = csr_matrix([[exp_encoded, loc_encoded]])

# Must exactly mirror preprocessing.py's hstack order
X_input = hstack([
    numeric_features,
    skills_tfidf,
    title_tfidf, title_tfidf, title_tfidf,   # 3x — same as training
    desc_tfidf,
    action_tfidf
])


# STEP 5 — Predict

prediction     = model.predict(X_input)[0]
probabilities  = model.predict_proba(X_input)[0]

predicted_industry = encoders['industry'].classes_[prediction]

# Get top 3 predictions with confidence
top3_indices = probabilities.argsort()[-3:][::-1]
top3 = [(encoders['industry'].classes_[i], round(probabilities[i]*100, 2))
        for i in top3_indices]

# STEP 6 — Show Results

print("\n" + "="*50)
print("   PREDICTION RESULTS")
print("="*50)
print(f"\n Predicted Industry: {predicted_industry}")
print(f"\n Top 3 Possibilities:")
for i, (industry, prob) in enumerate(top3, 1):
    bar = '' * int(prob // 5)
    print(f"   {i}. {industry:<35} {prob:>6.2f}%  {bar}")

print("\n" + "="*50)