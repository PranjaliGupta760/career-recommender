import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import hstack
import pickle 

#Load Dataset 
data=pd.read_csv('data/dataset_job_recommendation_system.csv')

print("Original dataset shape:", data.shape)

# STEP 1 (ENCODE CATEGORICAL VARIABLES)
# Make Encoder for the every column 

label_industry=LabelEncoder()
label_experience=LabelEncoder()
label_location=LabelEncoder()

# Apply Encoding for the columns Industry,experience and location

data['Industry_encoded']=label_industry.fit_transform(data['Industry'])
#fit_transform->Make vocabulary and check unique values and then convert it into numbers using transform
data['Experience_encoded']=label_experience.fit_transform(data['Experience Level'])
data['Location_encoded']=label_location.fit_transform(data['Location'])

print("\n Categorical variables encoded")
print(f"Industries: {list(label_industry.classes_)}")
print(f"Experience: {list(label_experience.classes_)}")
print(f"Locations: {list(label_location.classes_)}")

# Step-2 Apply TF-IDF for the columns skills and job_title
#Convert text skills to numbers

vectorizer=TfidfVectorizer(max_features=50,lowercase=True)
skills_tfidf=vectorizer.fit_transform(data['Required Skills'])

print(f"\n Step 2: Skills vectorized to {skills_tfidf.shape[1]} features ")
print(f"   All skills: {vectorizer.get_feature_names_out().tolist()}")

# Step-3 Combine All Features

numeric_features=data[['Salary','Industry_encoded','Experience_encoded','Location_encoded']].values

# Combine numeric+text features (create feature matrix)
X= hstack([numeric_features,skills_tfidf])
#  hstack = horizontal stack (side by side)

print(f"\n Step 3: Features Combined")
print(f"Total features:{X.shape[1]}")
print(f"Total Sample:{X.shape[0]}")

# Step-4 (Target Variable)

y=data['Industry_encoded'].values

print(f"\n Target variables ready")
print(f"Classes:{list(label_industry.classes_)}")

# Step 5 (Save for later use)
# save preprocessed data using pickle

with open('models/X_features.pkl','wb') as f:
    pickle.dump(X,f)

# Save X(feature matrix) in the file format in models/folder ( Binary form)

with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('models/y_target.pkl', 'wb') as f:
    pickle.dump(y, f)


with open('models/encoders.pkl', 'wb') as f:
    pickle.dump({
        'industry': label_industry, 
        'experience': label_experience, 
        'location': label_location
    }, f)

print("\n Step 5: All features saved!")
print("   Files saved in models/ folder")
print("\n Feature Engineering Complete! Ready for model training.")
