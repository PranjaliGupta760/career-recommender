import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pickle

# Load merged dataset
data = pd.read_csv('data/dataset_job_recommendation_system.csv')

print("Original dataset shape:", data.shape)

# ─────────────────────────────────────────
# STEP 1 — Encode Categorical Variables
# ─────────────────────────────────────────

label_experience = LabelEncoder()
label_location   = LabelEncoder()

# fit_transform: find unique values → convert to numbers
data['Experience_encoded'] = label_experience.fit_transform(data['Experience Level'])
data['Location_encoded']   = label_location.fit_transform(data['Location'])

print("\nStep 1: Categorical variables encoded")
print(f"  Experience Levels : {list(label_experience.classes_)}")
print(f"  Unique Locations  : {data['Location_encoded'].nunique()}")

# ─────────────────────────────────────────
# STEP 2 — TF-IDF on Skills
# ─────────────────────────────────────────
# Convert text skills into numbers using TF-IDF
# max_features=50 means keep top 50 most important skill words

vectorizer   = TfidfVectorizer(max_features=50, lowercase=True)
skills_tfidf = vectorizer.fit_transform(data['Required Skills'])

print(f"\nStep 2: Skills vectorized to {skills_tfidf.shape[1]} features")
print(f"  Skills: {vectorizer.get_feature_names_out().tolist()}")

# ─────────────────────────────────────────
# STEP 3 — Combine All Features
# ─────────────────────────────────────────
# NOTE: Industry is NOT included here (that is our target, not a feature)

numeric_features = data[['Salary',
                          'Experience_encoded',
                          'Location_encoded']].values

# hstack = horizontal stack (join side by side)
X = hstack([numeric_features, skills_tfidf])

print(f"\nStep 3: Features combined")
print(f"  Total features : {X.shape[1]}")
print(f"  Total samples  : {X.shape[0]}")

# ─────────────────────────────────────────
# STEP 4 — Target Variable (Industry)
# ─────────────────────────────────────────

label_industry = LabelEncoder()
y = label_industry.fit_transform(data['Industry'])

print(f"\nStep 4: Target variable ready")
print(f"  Industries ({len(label_industry.classes_)}): {list(label_industry.classes_)}")

# ─────────────────────────────────────────
# STEP 5 — Save for Model Training
# ─────────────────────────────────────────

with open('models/X_features.pkl', 'wb') as f:
    pickle.dump(X, f)

with open('models/y_target.pkl', 'wb') as f:
    pickle.dump(y, f)

with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('models/encoders.pkl', 'wb') as f:
    pickle.dump({
        'industry'  : label_industry,
        'experience': label_experience,
        'location'  : label_location
    }, f)

print("\nStep 5: All files saved in models/ folder")
print("\n✅ Preprocessing complete! Ready for model training.")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")