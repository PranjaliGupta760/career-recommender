import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading Preprocessed data....")

# Load saved features and target 
with open('models/X_features.pkl','rb') as f:
    X=pickle.load(f)

with open('models/y_target.pkl','rb') as f:
    y=pickle.load(f)

with open('models/encoders.pkl','rb') as f:
    encoders=pickle.load(f)

print(f"Data loaded: X Shape {X.shape}, y shape {y.shape}" )

# Step-1 : Split Data
#test-size=0.2 means 20% testing data and random_state 42 means equal split every time
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print(f"\nStep-1:Data Split....")
print(f"Training: {X_train.shape[0]} samples")
print(f"Testing:{X_test.shape[0]} smaples")

# Step-2 Train Logistic regression
print(f"\nStep-2:Train Logistic Regression....")
model=LogisticRegression(max_iter=5000)
model.fit(X_train,y_train)

print(f"Model Trained !")

# Step-3 Make predictions 

y_pred=model.predict(X_test)

print(f"\n predictions made on test set")

# Step-4 Evaluate Model
accuracy = accuracy_score(y_test,y_pred)

print(f"\n Model Evaluation")

print(f" Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\n Classification Report:")
print(classification_report(y_test,y_pred,target_names=encoders['industry'].classes_))

# Step-5 Save Model 
with open('models/trained_model.pkl','wb') as f:
    pickle.dump(model,f)

print(f"\n Step 5: Model saved!")

# Step-6 Confusion Matrix

cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True, fmt='d',cmap='Blues',xticklabels=encoders['industry'].classes_,yticklabels=encoders['industry'].classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.savefig('graphs/confusion_matrix.png')
plt.show()

print(f"\n Model Training Complete!")
print(f"   Model saved: models/trained_model.pkl")
print(f"   Confusion matrix: graphs/confusion_matrix.png")
