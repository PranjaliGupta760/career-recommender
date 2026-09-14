import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load merged dataset
data = pd.read_csv('data/dataset_job_recommendation_system.csv')

# ─────────────────────────────
# Graph 1 — Top 10 Job Titles
# ─────────────────────────────

top_jobs = data['Job Title'].value_counts().head(10)
plt.figure(figsize=(10, 6))
top_jobs.plot(kind='bar', color='blue')
plt.title("Top 10 Job Titles")
plt.xlabel('Job Title')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('graphs/top_jobs.png')
plt.show()

# ─────────────────────────────
# Graph 2 — Industry Distribution
# ─────────────────────────────

plt.figure(figsize=(10, 8))
industry = data['Industry'].value_counts()
industry.plot(kind='pie', autopct='%1.1f%%')
# autopct shows percentage: 1 digit before decimal, 1 after
plt.title("Industry Distribution (Top 10)")
plt.tight_layout()
plt.savefig('graphs/industry.png')
plt.show()

# ─────────────────────────────
# Graph 3 — Salary Distribution
# ─────────────────────────────

plt.figure(figsize=(10, 5))
sns.histplot(data['Salary'], bins=30, color='teal')
plt.title('Salary Distribution')
plt.xlabel('Salary (USD/year)')
plt.tight_layout()
plt.savefig('graphs/salary.png')
plt.show()

# ─────────────────────────────
# Graph 4 — Top 15 Skills
# ─────────────────────────────

all_skills = []
for skills in data['Required Skills'].dropna():
    # dropna removes null values so strip does not give any error
    all_skills.extend(s.strip() for s in skills.split(','))

skill_counts = Counter(all_skills)
top_skills   = pd.DataFrame(skill_counts.most_common(15),
                             columns=['Skill', 'Count'])

plt.figure(figsize=(10, 6))
sns.barplot(data=top_skills, x='Count', y='Skill', color='red')
plt.title('Top 15 High Demand Skills')
plt.tight_layout()
plt.savefig('graphs/skills.png')
plt.show()

# ─────────────────────────────
# Graph 5 — Experience Level Distribution (NEW)
# ─────────────────────────────

plt.figure(figsize=(8, 5))
exp_counts = data['Experience Level'].value_counts()
sns.barplot(x=exp_counts.index, y=exp_counts.values, color='purple')
plt.title('Jobs by Experience Level')
plt.xlabel('Experience Level')
plt.ylabel('Count')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('graphs/experience.png')
plt.show()

# ─────────────────────────────
# Graph 6 — Avg Salary by Industry (NEW)
# ─────────────────────────────

plt.figure(figsize=(12, 6))
avg_sal = data.groupby('Industry')['Salary'].mean().sort_values(ascending=False)
sns.barplot(x=avg_sal.values, y=avg_sal.index, color='orange')
plt.title('Average Salary by Industry')
plt.xlabel('Average Salary (USD/year)')
plt.tight_layout()
plt.savefig('graphs/salary_by_industry.png')
plt.show()

print("All graphs saved in graphs/ folder")