import pandas as pd

# Load merged dataset
data = pd.read_csv('data/dataset_job_recommendation_system.csv')

print("Shape:", data.shape)
# shape tells size of file in format (rows, columns)

print("\nColumns:", data.columns.tolist())
# columns tells the column names and tolist converts to list format

print("\nFirst 5 rows:")
print(data.head())
# head() returns first 5 rows of data

print("\nNull Values:")
print(data.isnull().sum())
# returns count of null values in each column

print("\nIndustry Value Count:")
print(data['Industry'].value_counts())
# value_counts returns total count per industry

print("\nExperience Level Value Count:")
print(data['Experience Level'].value_counts())

# describe() provides statistics like mean, min, max, count, std

print("\nUnique Job Titles:", data['Job Title'].nunique())
print("Unique Locations:", data['Location'].nunique())