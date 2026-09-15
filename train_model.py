import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading Preprocessed data....")

with open('models/X_features.pkl', 'rb') as f:
    X = pickle.load(f)
with open('models/y_target.pkl', 'rb') as f:
    y = pickle.load(f)
with open('models/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

print(f"Data loaded: X Shape {X.shape}, y shape {y.shape}")

# Step 1 — First split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Then split train further into train + validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42)

print(f"\nStep-1: Data Split....")
print(f"  Training:   {X_tr.shape[0]} samples")
print(f"  Validation: {X_val.shape[0]} samples")
print(f"  Testing:    {X_test.shape[0]} samples")

# Step 2 — Train XGBoost
print(f"\nStep-2: Train XGBoost Algorithm....")

model = XGBClassifier(
    max_depth=4,
    n_estimators=500,
    learning_rate=0.04,
    subsample=0.65,
    colsample_bytree=0.65,
    min_child_weight=7,
    reg_lambda=5.0,
    reg_alpha=1.0,
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],  # validation set for early stopping
    verbose=False
)

print(f"Model Trained!")
print(f"  Best iteration: {model.best_iteration}")  # shows when early stopping kicked in

# Step 3 — Predictions
y_pred = model.predict(X_test)
print(f"\nPredictions made on test set")

# Step 4 — Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Evaluation")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=encoders['industry'].classes_))

# Step 5 — Save Model
with open('models/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print(f"\nStep 5: Model saved!")

# Step 6 — Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoders['industry'].classes_,
            yticklabels=encoders['industry'].classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('graphs/confusion_matrix.png')
plt.show()

print(f"\nModel Training Complete!")
print(f"  Model saved: models/trained_model.pkl")
print(f"  Confusion matrix: graphs/confusion_matrix.png")

train_acc = model.score(X_train, y_train)
val_acc = model.score(X_val, y_val)
test_acc = model.score(X_test, y_test)

print(f"Actual Training Accuracy: {train_acc*100:.2f}%")
print(f"Actual Validation Accuracy: {val_acc*100:.2f}%")
print(f"Actual Testing Accuracy:  {test_acc*100:.2f}%")
print(f"Real Overfitting Gap:    {(train_acc - test_acc)*100:.2f}%")