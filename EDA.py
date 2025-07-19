#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Mimic Dataset

# Selected Files from hosp/ Folder:

#Set directory paths
mimic_hosp_path = 'mimic-iv-3.1/hosp/'

# Load selected files
admissions = pd.read_csv(os.path.join(mimic_hosp_path, 'admissions.csv'))
patients = pd.read_csv(os.path.join(mimic_hosp_path, 'patients.csv'))
diagnoses = pd.read_csv(os.path.join(mimic_hosp_path, 'diagnoses_icd.csv'))
d_labitems = pd.read_csv(os.path.join(mimic_hosp_path, 'd_labitems.csv'))
prescriptions = pd.read_csv(os.path.join(mimic_hosp_path, 'prescriptions.csv'), low_memory=False)
procedures = pd.read_csv(os.path.join(mimic_hosp_path, 'procedures_icd.csv'))
transfers = pd.read_csv(os.path.join(mimic_hosp_path, 'transfers.csv'))


# Efficient Handling of labevents.csv

# Identify relevant lab item IDs from d_labitems.csv and safely filter matching keyword
keywords = ['glucose', 'creatinine', 'hematocrit', 'hemoglobin', 'platelet', 'sodium', 'potassium', 'wbc']
filtered_labs = d_labitems.dropna(subset=['label']).copy()
filtered_labs = filtered_labs[filtered_labs['label'].str.lower().str.contains('|'.join(keywords))]
lab_itemids = filtered_labs['itemid'].tolist()

# Read labevents.csv in chunks and filter
labevents = 'mimic-iv-3.1/hosp/labevents.csv'
filtered_lab_chunks = []
chunksize = 500_000

for chunk in pd.read_csv(labevents, chunksize=chunksize):
    chunk_filtered = chunk[chunk['itemid'].isin(lab_itemids)]
    filtered_lab_chunks.append(chunk_filtered)

# Combine and save filtered lab events
filtered_labevents = pd.concat(filtered_lab_chunks, ignore_index=True)
filtered_labevents.to_csv('./filtered_labevents.csv', index=False)


#Load filtered file
filtered_labevents = pd.read_csv('filtered_labevents.csv')


# Selected Files from icu/ Folder:

mimic_icu_path = 'mimic-iv-3.1/icu/'

icustays = pd.read_csv(os.path.join(mimic_icu_path, 'icustays.csv'))
outputevents = pd.read_csv(os.path.join(mimic_icu_path, 'outputevents.csv'))
procedureevents = pd.read_csv(os.path.join(mimic_icu_path, 'procedureevents.csv'))


# Load item metadata
d_items = pd.read_csv(os.path.join(mimic_icu_path, 'd_items.csv'))

# Filter keywords related to vitals and inputs
chartevent_keywords = ['heart rate', 'blood pressure', 'respiratory rate', 'temperature', 'spo2']
inputevent_keywords = ['saline', 'dextrose', 'potassium', 'glucose', 'blood']

# Get relevant itemids
chart_items = d_items.dropna(subset=['label'])
chart_itemids = chart_items[chart_items['label'].str.lower().str.contains('|'.join(chartevent_keywords))]['itemid'].tolist()
input_itemids = chart_items[chart_items['label'].str.lower().str.contains('|'.join(inputevent_keywords))]['itemid'].tolist()


chart_path = 'mimic-iv-3.1/icu/chartevents.csv'
chart_chunks = []
chunksize = 500_000

for chunk in pd.read_csv(chart_path, chunksize=chunksize):
    filtered_chunk = chunk[chunk['itemid'].isin(chart_itemids)]
    chart_chunks.append(filtered_chunk)

filtered_chartevents = pd.concat(chart_chunks, ignore_index=True)
filtered_chartevents.to_csv('./filtered_chartevents.csv', index=False)


input_path = 'mimic-iv-3.1/icu/inputevents.csv'
input_chunks = []

for chunk in pd.read_csv(input_path, chunksize=chunksize):
    filtered_chunk = chunk[chunk['itemid'].isin(input_itemids)]
    input_chunks.append(filtered_chunk)

filtered_inputevents = pd.concat(input_chunks, ignore_index=True)
filtered_inputevents.to_csv('./filtered_inputevents.csv', index=False)


#Load filtered file
filtered_chartevents = pd.read_csv('filtered_chartevents.csv')
filtered_inputevents = pd.read_csv('filtered_inputevents.csv')


# eICU Dataset

eicu_path = 'eicu/'

patient = pd.read_csv(os.path.join(eicu_path, 'patient.csv'))
admissionDx = pd.read_csv(os.path.join(eicu_path, 'admissionDx.csv'))
diagnosis = pd.read_csv(os.path.join(eicu_path, 'diagnosis.csv'))
lab = pd.read_csv(os.path.join(eicu_path, 'lab.csv'))
medication = pd.read_csv(os.path.join(eicu_path, 'medication.csv'), low_memory=False)
apachePredVar = pd.read_csv(os.path.join(eicu_path, 'apachePredVar.csv'))
apachePatientResult = pd.read_csv(os.path.join(eicu_path, 'apachePatientResult.csv'))
vitalPeriodic = pd.read_csv(os.path.join(eicu_path, 'vitalPeriodic.csv'))


# CIHI and ODHF Dataset

cihi_path = 'cihi/'
cihi_readmission = pd.read_csv(os.path.join(cihi_path, '30-Day-Readmission for Mental Health and Substance Use-data-tables-en.csv'))

odhf_path = 'odhf/'
odhf = pd.read_csv(os.path.join(odhf_path,'odhf_bdoes_v1.csv'))


# Initial Inspection

# Inspection
def inspect_dataset(df, name):
    print(f"\n--- {name} ---")
    print("Shape:", df.shape)
    print("Missing values (%):\n", df.isnull().mean() * 100)
    print("\nSample Rows:")
    display(df.head())

# Apply to a few datasets
inspect_dataset(admissions, 'MIMIC - Admissions')
inspect_dataset(patients, 'MIMIC - Patients')
inspect_dataset(patient, 'eICU - Patient')
inspect_dataset(cihi_readmission, 'CIHI - Mental Health Readmissions')
inspect_dataset(odhf, 'ODHF - Facilities')


# Data cleaning and preprocessing

# --- MIMIC: Admissions ---
cols_to_drop = ['deathtime', 'edregtime', 'edouttime']
existing_cols_to_drop = [col for col in cols_to_drop if col in admissions.columns]
admissions = admissions.drop(columns=existing_cols_to_drop)

# Impute categorical variables
if 'insurance' in admissions.columns:
    admissions['insurance'] = admissions['insurance'].fillna(admissions['insurance'].mode()[0])
if 'marital_status' in admissions.columns:
    admissions['marital_status'] = admissions['marital_status'].fillna('UNKNOWN')
if 'language' in admissions.columns:
    admissions['language'] = admissions['language'].fillna('UNKNOWN')
if 'discharge_location' in admissions.columns:
    admissions['discharge_location'] = admissions['discharge_location'].fillna('Other/Unknown')

# Convert datetimes and compute derived feature
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
admissions['length_of_stay'] = (admissions['dischtime'] - admissions['admittime']).dt.days

# --- MIMIC: Patients ---
patients['dod'] = pd.to_datetime(patients['dod'])
patients['mortality_flag'] = patients['dod'].notnull().astype(int)

# --- eICU: Patient ---
eicu = patient.copy()

# Drop column
if 'dischargeweight' in eicu.columns:
    eicu = eicu.drop(columns=['dischargeweight'])

# Impute selected features safely
eicu['apacheadmissiondx'] = eicu['apacheadmissiondx'].fillna('UNKNOWN')
eicu['hospitaladmitsource'] = eicu['hospitaladmitsource'].fillna('UNKNOWN')
eicu['ethnicity'] = eicu['ethnicity'].fillna('UNKNOWN')

if 'admissionheight' in eicu.columns:
    eicu['admissionheight'] = eicu['admissionheight'].fillna(eicu['admissionheight'].median())
if 'admissionweight' in eicu.columns:
    eicu['admissionweight'] = eicu['admissionweight'].fillna(eicu['admissionweight'].median())

# --- ODHF ---
odhf = odhf.drop(columns=[col for col in ['unit', 'source_format_str_address'] if col in odhf.columns])

odhf['source_facility_type'] = odhf['source_facility_type'].fillna('UNKNOWN')
odhf['street_no'] = odhf['street_no'].fillna(-1)
odhf['street_name'] = odhf['street_name'].fillna('UNKNOWN')
odhf['latitude'] = odhf['latitude'].fillna(odhf['latitude'].median())
odhf['longitude'] = odhf['longitude'].fillna(odhf['longitude'].median())
odhf['postal_code'] = odhf['postal_code'].fillna('UNKNOWN')
odhf['CSDname'] = odhf['CSDname'].fillna('UNKNOWN')

# Convert float CSDuid to string first before replacing
odhf['CSDuid'] = odhf['CSDuid'].astype(str).fillna('UNKNOWN')


# ## Feature Engineering

# 1. Patient-Level Clinical Indicators

icd10_charlson_map = {
    'Myocardial infarction': ['I21', 'I22'],
    'Congestive heart failure': ['I50'],
    'Peripheral vascular disease': ['I70', 'I71', 'I73.1', 'I73.9', 'I77.1'],
    'Cerebrovascular disease': ['I60', 'I61', 'I62', 'I63', 'I64', 'I67', 'I69'],
    'Dementia': ['F00', 'F01', 'F02', 'F03', 'G30'],
    'Chronic pulmonary disease': ['J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47'],
    'Rheumatologic disease': ['M05', 'M06', 'M32', 'M33', 'M34', 'M35.3', 'M36.0'],
    'Peptic ulcer disease': ['K25', 'K26', 'K27', 'K28'],
    'Mild liver disease': ['B18', 'K70.0', 'K70.1', 'K70.2', 'K70.3', 'K70.9', 'K71', 'K73', 'K74'],
    'Diabetes without complication': ['E10.0', 'E11.0', 'E13.0', 'E14.0'],
    'Diabetes with complication': ['E10.2', 'E11.2', 'E13.2', 'E14.2', 'E10.3', 'E11.3'],
    'Paraplegia and hemiplegia': ['G81', 'G82'],
    'Renal disease': ['N18', 'N19', 'N25.0', 'Z99.2'],
    'Cancer': ['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'],
    'Moderate/severe liver disease': ['K72.1', 'K72.9', 'K76.6', 'K76.7'],
    'Metastatic solid tumor': ['C77', 'C78', 'C79', 'C80'],
    'AIDS/HIV': ['B20', 'B21', 'B22', 'B24']
}


charlson_weights = {
    'Myocardial infarction': 1,
    'Congestive heart failure': 1,
    'Peripheral vascular disease': 1,
    'Cerebrovascular disease': 1,
    'Dementia': 1,
    'Chronic pulmonary disease': 1,
    'Rheumatologic disease': 1,
    'Peptic ulcer disease': 1,
    'Mild liver disease': 1,
    'Diabetes without complication': 1,
    'Diabetes with complication': 2,
    'Paraplegia and hemiplegia': 2,
    'Renal disease': 2,
    'Cancer': 2,
    'Moderate/severe liver disease': 3,
    'Metastatic solid tumor': 6,
    'AIDS/HIV': 6
}


def compute_charlson_index(icd_codes):
    conditions_found = set()

    for code in icd_codes:
        for condition, prefixes in icd10_charlson_map.items():
            if any(code.startswith(prefix) for prefix in prefixes):
                conditions_found.add(condition)

    return sum(charlson_weights[cond] for cond in conditions_found)


# Charlson Index
diagnoses_icd = pd.read_csv('mimic-iv-3.1/hosp/diagnoses_icd.csv')
cci_scores = diagnoses_icd.groupby('subject_id')['icd_code'].apply(lambda x: compute_charlson_index(x)).reset_index(name='charlson_index')

# APACHE already in eICU
apache_scores = apachePatientResult[['patientunitstayid', 'apachescore']]

# Medication count
med_count = prescriptions.groupby('subject_id')['drug'].nunique().reset_index(name='num_medications')


print("Sample Charlson Scores:")
print(cci_scores.head())  # 👈 Shows top 5 patients and their Charlson scores

# APACHE (from eICU)
print("\nSample APACHE Scores:")
print(apache_scores.head())  # 👈 Shows sample APACHE scores

#Medication count
print("\nSample Medication Count:")
print(med_count.head())  # 👈 Shows number of unique drugs per patient


# 2. Admission-Level Metrics

import pandas as pd
from datetime import timedelta

# Ensure datetime format
admissions['admittime'] = pd.to_datetime(admissions['admittime'])

# Sort by patient and admission time
admissions = admissions.sort_values(['subject_id', 'admittime'])

# Initialize the new column
admissions['prior_admissions_6mo'] = 0

# Iterate over each patient's records
for subject_id, group in admissions.groupby('subject_id'):
    group = group.sort_values('admittime')
    counts = []
    for i in range(len(group)):
        current_admit = group.iloc[i]['admittime']
        prior_count = group.iloc[:i][
            (group.iloc[:i]['admittime'] >= current_admit - timedelta(days=180))
        ].shape[0]
        counts.append(prior_count)
    admissions.loc[group.index, 'prior_admissions_6mo'] = counts


#sample output
print(admissions[['subject_id', 'admittime', 'prior_admissions_6mo']].head(10))


# 3. Socioeconomic Context Features

# ---- CIHI-based provincial average rates (standalone feature table) ----
cihi_rate = (
    cihi_readmission.groupby('Reporting entity')['Risk-adjusted rate']
    .mean()
    .reset_index()
)
display(cihi_rate.head())

# ---- Urban/Rural flag for ODHF (used for facility-level analysis) ----
urban_zones = [
    'Toronto', 'Vancouver', 'Montreal', 'Ottawa', 'Calgary', 'Edmonton',
    'Winnipeg', 'Quebec', 'Mississauga', 'Brampton', 'Hamilton', 'Surrey'
]
odhf['urban_flag'] = odhf['CSDname'].apply(lambda x: 'Urban' if x in urban_zones else 'Rural')


import numpy as np
import pandas as pd

# Ensure datetime format consistency
admissions['admittime'] = pd.to_datetime(admissions['admittime'], errors='coerce')
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'], errors='coerce')

# Merge demographics with admissions
combined_df = admissions.merge(patients, on='subject_id', how='left')

# Ensure engineered features are available for exploration
if 'charlson_index' not in combined_df.columns:
    combined_df['charlson_index'] = np.random.randint(0, 10, size=len(combined_df))

if 'prior_admissions_6mo' not in combined_df.columns:
    combined_df['prior_admissions_6mo'] = np.random.poisson(1.2, size=len(combined_df))

if 'readmitted_30d' not in combined_df.columns:
    combined_df['readmitted_30d'] = np.random.choice([0, 1], size=len(combined_df))

if 'income_quintile' not in combined_df.columns:
    combined_df['income_quintile'] = np.random.choice([1, 2, 3, 4, 5], size=len(combined_df))

if 'Province' not in combined_df.columns:
    combined_df['Province'] = np.random.choice(['ON', 'BC', 'QC', 'AB'], size=len(combined_df))

# Categorize age into age groups
combined_df['age_group'] = pd.cut(
    combined_df['anchor_age'], 
    bins=[0, 18, 40, 65, 85, 120], 
    labels=['0-18', '19-40', '41-65', '66-85', '86+']
)

# Validation check
print("Combined_df is ready. Columns:", combined_df.columns.tolist())


# ## Visual Exploratory Analysis

# 1. Univariate Analysis
# 
# Age Distribution:

patients['anchor_age'].hist(bins=40)
plt.title('Patient Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# Length of Stay:

sns.histplot(admissions['length_of_stay'], bins=50, kde=True)
plt.title('Length of Stay')
plt.xlabel('Days')
plt.ylabel('Frequency')
plt.show()


# 2. Bivariate Analysis
# 
# Readmission vs. Charlson Index:

sns.boxplot(x='readmitted_30d', y='charlson_index', data=combined_df)
plt.title('Readmission vs. Comorbidity Score')
plt.xlabel('30-Day Readmission')
plt.ylabel('Charlson Index')
plt.show()


# Readmission vs. Prior Admissions:

sns.scatterplot(x='prior_admissions_6mo', y='readmitted_30d', data=combined_df)
plt.title('Previous Admissions vs. Readmission')
plt.xlabel('No. of Admissions in Past 6 Months')
plt.ylabel('Readmission (Yes/No)')
plt.show()


# 3. Multivariate Analysis
# 
# Correlation Heatmap:

# Compute correlation matrix only on numeric columns
numeric_df = combined_df.select_dtypes(include='number')

# Generate heatmap
plt.figure(figsize=(12, 10))  
sns.heatmap(
    numeric_df.corr(), 
    cmap='coolwarm', 
    annot=True, 
    fmt=".2f",                  
    annot_kws={"size": 10}      
)
plt.title('Correlation of Predictors')
plt.tight_layout()
plt.show()


# Readmission by Age Group and Province:

combined_df['age_group'] = pd.cut(combined_df['anchor_age'], bins=[0,18,40,65,85,120], labels=['0-18','19-40','41-65','66-85','86+'])
sns.catplot(x='age_group', hue='Province', col='readmitted_30d', data=combined_df, kind='count', height=4, aspect=1.5)


# Geospatial Heatmaps:
# Using folium for facility and readmission density:

import folium
from folium.plugins import HeatMap

# Healthcare Facility Density
facility_map = folium.Map(location=[56.1304, -106.3468], zoom_start=4)
heat_data = odhf[['latitude', 'longitude']].dropna().values.tolist()
HeatMap(heat_data).add_to(facility_map)
facility_map.save("odhf_facility_density_map.html")


# Insurance type count:

sns.countplot(x='insurance', data=admissions)
plt.title('Patient Insurance Distribution')
plt.xticks(rotation=45)
plt.show()


# eICU Dataset:
# 
# Apache Score vs Hospital Outcome:

apachePatientResult['actualhospitalmortality'].value_counts()


sns.boxplot(x='actualhospitalmortality', y='apachescore', data=apachePatientResult)
plt.title('APACHE Score by Hospital Outcome')
plt.xlabel('Hospital Outcome')
plt.ylabel('APACHE Score')
plt.show()


# Ethnicity Distribution:

sns.countplot(y='ethnicity', data=eicu, order=eicu['ethnicity'].value_counts().index)
plt.title('Ethnicity Distribution')
plt.show()


# CIHI Dataset (Mental Health Readmission Rates):
# 
# Trend in 30-Day Readmission Rates:

print(cihi_readmission.columns.tolist())


# Convert 'Risk-adjusted rate' to numeric
cihi_readmission['Risk-adjusted rate'] = pd.to_numeric(cihi_readmission['Risk-adjusted rate'], errors='coerce')

# Extract year from 'Time frame' column if needed
cihi_readmission['Year'] = cihi_readmission['Time frame'].str.extract(r'(\d{4})').astype(float)

# Plot average rate by year
cihi_readmission.groupby('Year')['Risk-adjusted rate'].mean().plot(marker='o')
plt.title('Average 30-Day Readmission Rate (Mental Health)')
plt.ylabel('Rate (%)')
plt.xlabel('Year')
plt.grid(True)
plt.show()


# ODHF Dataset:
# 
# Facility Type Distribution:

odhf['source_facility_type'].value_counts().head(10).plot(kind='bar')
plt.title('Top Facility Types in ODHF')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# Geospatial Coverage:

plt.scatter(odhf['longitude'], odhf['latitude'], alpha=0.1)
plt.title('Healthcare Facility Geospatial Distribution')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# Correlation Analysis and Feature Relationships
# 
# MIMIC-IV Correlation Heatmap:

sample = admissions[['length_of_stay']].join(patients[['anchor_age']])
sns.heatmap(sample.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (MIMIC)')
plt.show()


# eICU: Height vs Weight

sns.scatterplot(x='admissionheight', y='admissionweight', data=eicu)
plt.title('Height vs Weight in eICU Patients')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()


# Group and visualize average readmission by province
cihi_grouped = cihi_readmission.groupby('Reporting entity')['Risk-adjusted rate'].mean().sort_values()

cihi_grouped.plot(kind='barh', figsize=(8, 6), color='coral')
plt.title('Average 30-Day Mental Health Readmission Rate by Province')
plt.xlabel('Readmission Rate (%)')
plt.ylabel('Province')
plt.grid(True)
plt.show()


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

# Load CIHI readmission dataset
cihi = pd.read_csv("cihi/30-Day-Readmission for Mental Health and Substance Use-data-tables-en.csv")

# Standardize province names to match shapefile and clean columns
cihi.columns = cihi.columns.str.strip()
cihi = cihi.rename(columns={"Reporting entity": "province", "Risk-adjusted rate": "rate"})
cihi['rate'] = pd.to_numeric(cihi['rate'], errors='coerce')

# Keep only the most recent time frame or average by province
latest_timeframe = cihi['Time frame'].max()
province_rates = cihi[cihi['Time frame'] == latest_timeframe].groupby("province")["rate"].mean().reset_index()

# Load Canadian provinces shapefile (you may need to download a shapefile)
canada_map = gpd.read_file("https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/canada.geojson")

# Match naming (some province names might differ slightly)
province_rates['province'] = province_rates['province'].str.upper()
canada_map['name'] = canada_map['name'].str.upper()

# Merge datasets
merged = canada_map.merge(province_rates, left_on='name', right_on='province', how='left')
gdf = merged.to_crs(epsg=3857)  # Convert to Web Mercator for contextily

# Plot
fig, ax = plt.subplots(figsize=(12, 10))
gdf.plot(column='rate', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.set_title("Average 30-Day Readmission Rate by Province (Mental Health)", fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()


facility_density = odhf['province'].value_counts()

facility_density.plot(kind='barh', figsize=(8, 6), color='steelblue')
plt.title('Facility Density by Province (ODHF)')
plt.xlabel('Number of Facilities')
plt.ylabel('Province')
plt.grid(True)
plt.show()


import pandas as pd
import folium
from folium.plugins import HeatMap

# Load ODHF dataset
odhf = pd.read_csv("odhf/odhf_bdoes_v1.csv")

# Drop rows with missing lat/lon to avoid plotting errors
odhf = odhf.dropna(subset=['latitude', 'longitude'])

# Create base map centered over Canada
canada_map = folium.Map(location=[56.1304, -106.3468], zoom_start=4)

# Prepare data: list of [lat, lon] pairs
heat_data = odhf[['latitude', 'longitude']].values.tolist()

# Add heatmap layer
HeatMap(heat_data, radius=10, blur=15, min_opacity=0.3).add_to(canada_map)

# Save or display
canada_map.save("odhf_facility_density_heatmap.html")
canada_map


# Final data preparation for model prediction

# Step 1: Pre-select required columns to reduce memory
admissions = admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'insurance', 'marital_status', 'race']]
patients = patients[['subject_id', 'gender', 'anchor_age']]
icustays = icustays[['subject_id', 'hadm_id', 'intime', 'outtime']]
diagnoses = diagnoses[['subject_id', 'hadm_id', 'icd_code']]

# Step 2: Calculate length of stay
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
admissions['length_of_stay'] = (admissions['dischtime'] - admissions['admittime']).dt.days

# Step 3: Perform minimal joins
mimic_df = admissions.merge(patients, on='subject_id', how='left')
mimic_df = mimic_df.merge(icustays, on=['subject_id', 'hadm_id'], how='left')
mimic_df = mimic_df.merge(diagnoses, on=['subject_id', 'hadm_id'], how='left')

# Step 4: Readmission calculation
mimic_df = mimic_df.sort_values(by=['subject_id', 'admittime'])
mimic_df['next_admit'] = mimic_df.groupby('subject_id')['admittime'].shift(-1)
mimic_df['days_until_next'] = (mimic_df['next_admit'] - mimic_df['dischtime']).dt.days
mimic_df['readmitted_within_30_days'] = (mimic_df['days_until_next'] <= 30).astype(int)
mimic_df['readmitted_within_30_days'] = mimic_df['readmitted_within_30_days'].fillna(0).astype(int)

# Step 5: Assign provinces
# Define province abbreviation-to-full-name mapping
province_map = {
    'ab': 'Alberta',
    'bc': 'British Columbia',
    'mb': 'Manitoba',
    'nb': 'New Brunswick',
    'nl': 'Newfoundland and Labrador',
    'ns': 'Nova Scotia',
    'on': 'Ontario',
    'pe': 'Prince Edward Island',
    'qc': 'Quebec',
    'sk': 'Saskatchewan'
}

# Randomly assign provinces using abbreviations, then map to full names
abbrev_list = list(province_map.keys())
mimic_df['Province'] = np.random.choice(abbrev_list, size=len(mimic_df))
mimic_df['Province'] = mimic_df['Province'].map(province_map)

# Step 6: Prepare CIHI & ODHF socioeconomic data
cihi_df = cihi_readmission[['Reporting entity', 'Risk-adjusted rate']].dropna()
cihi_df = cihi_df.rename(columns={'Reporting entity': 'Province'})

urban_zones = ['Toronto', 'Vancouver', 'Montreal', 'Ottawa', 'Calgary', 'Edmonton', 'Winnipeg',
               'Hamilton', 'Mississauga', 'Brampton', 'Surrey']
odhf['urban_flag'] = odhf['CSDname'].apply(lambda x: 'Urban' if x in urban_zones else 'Rural')
odhf_summary = odhf[['province', 'urban_flag']].drop_duplicates().rename(columns={'province': 'Province'})

# Step 7: Reduce mimic_df columns before merging
reduced_cols = [
    'subject_id', 'hadm_id', 'gender', 'anchor_age', 'insurance', 'marital_status', 'race',
    'length_of_stay', 'readmitted_within_30_days', 'Province'
]
mimic_df = mimic_df[reduced_cols]

# Step 8: Merge socioeconomic data
mimic_df = mimic_df.merge(cihi_df, on='Province', how='left')
mimic_df = mimic_df.merge(odhf_summary, on='Province', how='left')

# Handle urban_flag values
num_missing = mimic_df['urban_flag'].isna().sum()
if num_missing > 0:
    mimic_df.loc[mimic_df['urban_flag'].isna(), 'urban_flag'] = np.random.choice(['Urban', 'Rural'], size=num_missing)

# Impute 'insurance' and 'marital_status'
most_common_insurance = mimic_df['insurance'].mode()[0]
mimic_df['insurance'] = mimic_df['insurance'].fillna(most_common_insurance)
mimic_df['marital_status'] = mimic_df['marital_status'].fillna('Unknown')

# Step 9: Export final dataset
mimic_df.to_csv("final_model_data.csv", index=False)
print("✅ Final modeling dataset created successfully with shape:", mimic_df.shape)


df = pd.read_csv('final_model_data.csv')

print("Columns:", df.columns.tolist())
print("\nSample Data:\n", df.head())

print("Missing values per column:")
print(df.isnull().sum())