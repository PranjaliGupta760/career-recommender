import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
import pickle

# Load merged dataset
data = pd.read_csv('data/dataset_job_recommendation_system.csv')
print("Original dataset shape:", data.shape)

# ─────────────────────────────────────────
# STEP 0 — Action Phrase Extraction (NEW)
# ─────────────────────────────────────────
# Extracts only sentences where the PERSON acts — filters out
# "we're hiring for X" noise that causes industry bleed.

ACTION_VERBS = re.compile(
    r'\b(design|build|develop|manage|lead|analyze|create|implement|'
    r'source|screen|recruit|hire|interview|partner|coordinate|'
    r'administer|provide|monitor|assess|conduct|maintain|optimize|'
    r'collaborate|integrate|deploy|architect|deliver|support|write|'
    r'research|plan|execute|review|own|drive|define|establish)\b',
    re.IGNORECASE
)

def extract_action_phrases(text):
    if not isinstance(text, str):
        return ''
    sentences = re.split(r'[.!?\n]', text)
    action_sentences = [
        s.strip() for s in sentences
        if ACTION_VERBS.search(s.strip())
    ]
    return ' '.join(action_sentences) if action_sentences else text

data['action_phrases'] = data['description'].fillna('').apply(extract_action_phrases)

print("Step 0: Action phrases extracted")
print(f"  Sample: {data['action_phrases'].iloc[0][:120]}...")

# ─────────────────────────────────────────
# STEP 1 — Encode Categorical Variables
# ─────────────────────────────────────────

label_experience = LabelEncoder()
label_location   = LabelEncoder()

data['Experience_encoded'] = label_experience.fit_transform(data['Experience Level'])
data['Location']           = data['Location'].str.split(',').str[-1].str.strip()
data['Location_encoded']   = label_location.fit_transform(data['Location'])

print("\nStep 1: Categorical variables encoded")
print(f"  Experience Levels : {list(label_experience.classes_)}")
print(f"  Unique Locations  : {data['Location_encoded'].nunique()}")

# ─────────────────────────────────────────
# STEP 2 — TF-IDF Vectorizers (UPDATED)
# ─────────────────────────────────────────

# Job Title — bigrams added so "Technical Recruiter" and
# "Software Engineer" stay as single tokens, not split words
title_vectorizer = TfidfVectorizer(
    max_features=150,
    lowercase=True,
    ngram_range=(1, 2),      # NEW — bigrams
    analyzer='word'
)
title_tfidf = title_vectorizer.fit_transform(data['Job Title'])

# Skills — bigrams so "Talent Acquisition", "Machine Learning" stay intact
skills_vectorizer = TfidfVectorizer(
    max_features=100,
    lowercase=True,
    ngram_range=(1, 2),      # NEW — bigrams
    analyzer='word'
)
skills_tfidf = skills_vectorizer.fit_transform(data['Required Skills'])

# Full description — kept for broad context
desc_vectorizer = TfidfVectorizer(
    max_features=200,
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2)       # NEW — bigrams
)
desc_tfidf = desc_vectorizer.fit_transform(data['description'].fillna(''))

# Action phrases — captures role-specific verbs, not mentioned job titles
action_vectorizer = TfidfVectorizer(
    max_features=100,
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2)
)
action_tfidf = action_vectorizer.fit_transform(data['action_phrases'])

print(f"\nStep 2: Vectorization complete")
print(f"  Title features      : {title_tfidf.shape[1]}")
print(f"  Skills features     : {skills_tfidf.shape[1]}")
print(f"  Description features: {desc_tfidf.shape[1]}")
print(f"  Action features     : {action_tfidf.shape[1]}")

# ─────────────────────────────────────────
# STEP 3 — Combine Features (UPDATED)
# ─────────────────────────────────────────
# Title is tripled to boost its influence — it's the clearest
# signal of what industry a role belongs to.

numeric_features = data[['Experience_encoded', 'Location_encoded']].values

X = hstack([
    csr_matrix(numeric_features),
    skills_tfidf,
    title_tfidf, title_tfidf, title_tfidf,   # 3x weight boost
    desc_tfidf,
    action_tfidf                               # NEW — role-action signal
])

print(f"\nStep 3: Features combined")
print(f"  Total features : {X.shape[1]}")
print(f"  Total samples  : {X.shape[0]}")

# ─────────────────────────────────────────
# STEP 4 — Target Variable
# ─────────────────────────────────────────

label_industry = LabelEncoder()
y = label_industry.fit_transform(data['Industry'])

print(f"\nStep 4: Target variable ready")
print(f"  Industries ({len(label_industry.classes_)}): {list(label_industry.classes_)}")

# ─────────────────────────────────────────
# STEP 5 — Save Everything
# ─────────────────────────────────────────

with open('models/X_features.pkl', 'wb') as f:
    pickle.dump(X, f)

with open('models/y_target.pkl', 'wb') as f:
    pickle.dump(y, f)

with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(skills_vectorizer, f)        # renamed from vectorizer

with open('models/title_vectorizer.pkl', 'wb') as f:
    pickle.dump(title_vectorizer, f)

with open('models/desc_vectorizer.pkl', 'wb') as f:
    pickle.dump(desc_vectorizer, f)

with open('models/action_vectorizer.pkl', 'wb') as f:  # NEW
    pickle.dump(action_vectorizer, f)

with open('models/encoders.pkl', 'wb') as f:
    pickle.dump({
        'industry'  : label_industry,
        'experience': label_experience,
        'location'  : label_location
    }, f)

print("\nStep 5: All files saved")
print("\n✅ Preprocessing complete!")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")