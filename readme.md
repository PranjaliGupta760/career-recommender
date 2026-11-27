## Career Recommendation System

> Predicts suitable industries from a user profile using a trained ML model.

##  Features
- Merges and cleans multiple datasets (EDA, feature encoding)
- Trains an XGBoost classifier with regularization and early stopping
- Returns top 3 recommended industries with match probabilities
- Streamlit web interface for interactive predictions
- Evaluated using confusion matrices and per-class precision/recall

##  Tech Stack
- Python
- scikit-learn
- XGBoost
- Pandas
- Streamlit

##  How to Run
pip install -r requirements.txt
streamlit run predict.py

##  Project Structure
- `merge_data.py` – combines and merges source datasets
- `eda.py` / `explore.py` – exploratory data analysis
- `preprocessing.py` – data cleaning and feature encoding
- `train_model.py` – trains and saves the XGBoost model
- `predict.py` – Streamlit app for generating predictions
- `models/` – saved trained model(s)
- `graphs/` – evaluation plots (confusion matrix, precision/recall)

##  Use Case
Built to help users identify industries that match their profile, 
trained and validated on a merged, cleaned dataset with ~80% test accuracy.