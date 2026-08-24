import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter  

#Load dataset 
data=pd.read_csv('data/dataset_job_recommendation_system.csv')

# Graph 1 (Top 10 Job Titles)

top_jobs=data['Job Title'].value_counts().head(10) #Top 10 job titles by head() method
plt.figure(figsize=(10,6))
top_jobs.plot(kind='bar',color='blue')
plt.title("Top 10 job titles")
plt.xlabel('job title')
plt.ylabel('count')
plt.xticks(rotation=45,ha='right') # rotate x labels by 45 degree and in right allignment
plt.tight_layout() # tight_layout give proper spacing in graph
plt.savefig('graphs/top_jobs.png')
plt.show()

# Graph 2(Industry distribution)

plt.figure(figsize=(8,5))
industry=data['Industry'].value_counts()
industry.plot(kind='pie',autopct='%1.1f%%') #automatic percantage (%1.1f-> 1 digit before decimal and 1 digit after decimal and %% sign is to show % sign)
plt.title("Industry Distributions")
plt.tight_layout()
plt.savefig('graphs/industry.png')
plt.show()

# Graph 3 (salary distribution)
plt.figure(figsize=(10,5))
sns.histplot(data['Salary'],bins=30,color='teal')
plt.title('Salary Distribution')
plt.xlabel('Salary($)')
plt.tight_layout()
plt.savefig('graphs/salary.png')
plt.show()

# Graph 4 (Top 15 Skills)

all_skills=[]
for skills in data['Required Skills'].dropna():  # drpna remove null values so strip does not give any error 
    all_skills.extend(s.strip() for s in skills.split(','))

skill_counts=Counter(all_skills)
top_skills=pd.DataFrame(skill_counts.most_common(15), columns=['Skill','Count'])
plt.figure(figsize=(10,6))
sns.barplot(data=top_skills,x='Count',y='Skill',color='red')
plt.title('Top 15 High Demand Skills')
plt.tight_layout()
plt.savefig('graphs/skills.png')
plt.show()

print("All graphs saved in graph folder")
