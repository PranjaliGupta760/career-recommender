import pandas as pd

# ─────────────────────────────────────────
# STEP 1 — Load all CSV files
# ─────────────────────────────────────────

print("Loading CSV files...")

postings   = pd.read_csv('data/postings.csv',
                         usecols=['job_id','title','location',
                                  'formatted_experience_level'])

job_skills   = pd.read_csv('data/jobs/job_skills.csv')        # job_id, skill_abr
skills_map   = pd.read_csv('data/mappings/skills.csv')        # skill_abr, skill_name
job_ind      = pd.read_csv('data/jobs/job_industries.csv')    # job_id, industry_id
ind_map      = pd.read_csv('data/mappings/industries.csv')    # industry_id, industry_name
salaries     = pd.read_csv('data/jobs/salaries.csv',
                           usecols=['job_id','max_salary',
                                    'pay_period'])             # salary info

print(f"  Postings:      {postings.shape[0]:,} rows")
print(f"  Job Skills:    {job_skills.shape[0]:,} rows")
print(f"  Job Industries:{job_ind.shape[0]:,} rows")
print(f"  Salaries:      {salaries.shape[0]:,} rows")

# ─────────────────────────────────────────
# STEP 2 — Decode skills (abbr → name)
#          Group all skills per job into one string
# ─────────────────────────────────────────

skills_named = job_skills.merge(skills_map, on='skill_abr', how='left')

# Group: one row per job, skills as comma-separated string
skills_grouped = (skills_named
                  .groupby('job_id')['skill_name']
                  .apply(lambda x: ', '.join(x.dropna()))
                  .reset_index()
                  .rename(columns={'skill_name': 'Required Skills'}))

print(f"\n  Skills grouped: {skills_grouped.shape[0]:,} unique jobs")

# ─────────────────────────────────────────
# STEP 3 — Decode industries (id → name)
#          Keep one industry per job (first match)
# ─────────────────────────────────────────

ind_named = job_ind.merge(ind_map, on='industry_id', how='left')

# One industry per job
ind_per_job = (ind_named
               .groupby('job_id')['industry_name']
               .first()
               .reset_index()
               .rename(columns={'industry_name': 'Industry'}))

print(f"  Industries mapped: {ind_per_job.shape[0]:,} unique jobs")
print(f"  Total unique industries: {ind_per_job['Industry'].nunique()}")

# ─────────────────────────────────────────
# STEP 4 — Clean Salary
#          Use only YEARLY salaries
# ─────────────────────────────────────────

yearly_sal = salaries[salaries['pay_period'] == 'YEARLY'].copy()

# One salary row per job
salary_per_job = (yearly_sal
                  .groupby('job_id')['max_salary']
                  .first()
                  .reset_index()
                  .rename(columns={'max_salary': 'Salary'}))

print(f"  Yearly salaries found: {salary_per_job.shape[0]:,} jobs")

# ─────────────────────────────────────────
# STEP 5 — Merge everything into one dataset
# ─────────────────────────────────────────

data = postings.copy()
data = data.merge(skills_grouped, on='job_id', how='left')
data = data.merge(ind_per_job,    on='job_id', how='left')
data = data.merge(salary_per_job, on='job_id', how='left')

# Rename columns to match original code structure
data = data.rename(columns={
    'title'                      : 'Job Title',
    'location'                   : 'Location',
    'formatted_experience_level' : 'Experience Level'
})

print(f"\n  Merged dataset shape: {data.shape}")

# ─────────────────────────────────────────
# STEP 6 — Filter & Clean
# ─────────────────────────────────────────

# Drop rows missing key columns
before = len(data)
data = data.dropna(subset=['Industry', 'Required Skills',
                            'Experience Level', 'Salary'])
print(f"  Dropped {before - len(data):,} rows with missing values")
print(f"  Remaining rows: {len(data):,}")

# Keep only TOP 10 industries (422 is too many for a student project)
top10 = data['Industry'].value_counts().head(10).index
data  = data[data['Industry'].isin(top10)]
print(f"  After keeping top 10 industries: {len(data):,} rows")
print(f"\n  Industries kept:")
print(data['Industry'].value_counts())

# ─────────────────────────────────────────
# STEP 7 — Save merged dataset
# ─────────────────────────────────────────

data.to_csv('data/dataset_job_recommendation_system.csv', index=False)
print(f"\n✅ Saved: data/dataset_job_recommendation_system.csv")
print(f"   Final shape: {data.shape}")
print(f"\n   Columns: {data.columns.tolist()}")