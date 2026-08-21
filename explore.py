import pandas as pd 
#read_csv is the python function to read any csv file(excel file)
data=pd.read_csv('data/dataset_job_recommendation_system.csv')

print("Shape:", data.shape) 
#shape is used to tell the size of the file in the format (rows,columns)
print("\nColumns:",data.columns.tolist())
 # columns tell the column names and tolist convert in to list format
print("\nFirst 5 rows:",data.head())
# head() returns first 5 rows of the data

print("\nNull Values:",data.isnull().sum())
# returns the values that is null and count the total rows that data is null in a column

print("\nIndustry Value count:",data['Industry'].value_counts())
print("\nIndustry Value count:",data['Experience Level'].value_counts())
# value_counts returns total value of one industry like software : 3456 (3456 people mention software as thier industry)

print("\nSalary:",data['Salary'].describe())
#describe() provides whole statistics like mean,min,max,count,50%,std(which values are spread)