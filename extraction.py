
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


admissions=pd.read_csv('readmission/admissions_202208161605.csv')
cptevents=pd.read_csv('readmission/cptevents_202208161605.csv')
d_labitems=pd.read_csv('readmission/d_labitems_202208161605.csv')
diagnoses_icd=pd.read_csv('readmission/diagnoses_icd_202208161605.csv')
drgcodes=pd.read_csv('readmission/drgcodes_202208161605.csv')
labevents=pd.read_csv('readmission/labevents_202208161605.csv')
patients=pd.read_csv('readmission/patients_202208161605.csv')
procedures_icd=pd.read_csv('readmission/procedures_icd_202208161605.csv')

heart_failure_icd9 = [
    '39891','40201','40211','40291','40401','40403','40411','40413',
    '40491','40493','4280','4281','42820','42821','42822','42823',
    '42830','42831','42832','42833','42840','42841','42842','42843','4289'
]

hf_diagnoses = diagnoses_icd[diagnoses_icd['icd9_code'].isin(heart_failure_icd9)]

hf_admissions = pd.merge(hf_diagnoses, admissions, on='hadm_id', how='inner')
hf_admissions = hf_admissions.drop(['row_id_x', 'subject_id_x', 'row_id_y','ethnicity','marital_status','religion','language'], axis=1)

hf_admissions = hf_admissions.rename(columns={'subject_id_y': 'subject_id'})

hf_admissions['admittime'] = pd.to_datetime(hf_admissions['admittime'])
hf_admissions['dischtime'] = pd.to_datetime(hf_admissions['dischtime'])
hf_admissions['next_admit'] = hf_admissions.groupby('subject_id')['admittime'].shift(-1)
hf_admissions['days_to_readmit'] = (hf_admissions['next_admit'] - hf_admissions['dischtime']).dt.days
hf_admissions.loc[hf_admissions['days_to_readmit'] <= 0, 'days_to_readmit'] = pd.NA
hf_admissions['readmitted_30'] = hf_admissions['days_to_readmit'].apply(lambda x: 1 if pd.notna(x) and x <= 30 else 0)

drg_agg = drgcodes.groupby('hadm_id').agg({
    'drg_code': lambda x: ','.join(x.dropna().astype(str)),
    'description': lambda x: '; '.join(x.dropna().unique())
}).reset_index()

merged_df= hf_admissions.merge(drg_agg, on='hadm_id', how='left')

patients_new = patients[['subject_id', 'gender','expire_flag']]

labevents_new=labevents[['subject_id','itemid','charttime','value','valuenum','valueuom','flag']]
hf_diagnoses_1 = hf_diagnoses.drop(['row_id','seq_num'],axis=1)
lab_events_filtered = labevents_new[labevents_new['subject_id'].isin(hf_diagnoses_1['subject_id'])]

lab_events_filtered= pd.merge(lab_events_filtered, d_labitems, on='itemid', how='left')
lab_events_filtered=lab_events_filtered.drop(['row_id'],axis=1)
lab_events_filtered=pd.merge(lab_events_filtered, patients_new, on='subject_id', how='left')
lab_events_filtered=lab_events_filtered.drop(['itemid'],axis=1)
merged_df[merged_df.duplicated(subset='hadm_id', keep=False)].sort_values('hadm_id').head(10)
merged_df_1=merged_df
lab_events_filtered = lab_events_filtered[[
    'subject_id', 'charttime', 'valuenum', 'label', 'flag'
]]

merged_df = merged_df[[
    'hadm_id', 'subject_id', 'admittime', 'dischtime', 
    'admission_type', 'admission_location', 'discharge_location',
    'insurance', 'diagnosis', 'hospital_expire_flag', 
    'readmitted_30', 'days_to_readmit'
]]

hf_critical_labs = [
    'BNP', 'NTproBNP', 'Troponin', 'Creatinine', 'Sodium', 'Potassium',
    'Urea Nitrogen', 'BUN', 'Hemoglobin', 'Hematocrit', 'Albumin',
    'Glucose', 'Magnesium', 'Chloride', 'AST', 'ALT', 'Bilirubin',
    'C-Reactive Protein', 'CRP', 'D-Dimer', 'pH', 'pCO2', 'Lactate'
]

hf_critical_labs_lower = [x.lower() for x in hf_critical_labs]

filtered = lab_events_filtered[
    lab_events_filtered['label'].str.lower().isin(hf_critical_labs_lower)
].copy()

filtered['label'] = filtered['label'].str.lower() 

lab_wide = filtered.pivot_table(
    index=['subject_id', 'charttime'],
    columns='label',
    values='valuenum',
    aggfunc='first' 
).reset_index()

flag_wide = filtered.pivot_table(
    index=['subject_id', 'charttime'],
    columns='label',
    values='flag',
    aggfunc='first'
).reset_index()


flag_wide.columns = ['subject_id', 'charttime'] + [f"{col}_flag" for col in flag_wide.columns[2:]]


lab_wide['charttime'] = pd.to_datetime(lab_wide['charttime'])
flag_wide['charttime'] = pd.to_datetime(flag_wide['charttime'])

final_df = lab_wide.merge(flag_wide, on=['subject_id', 'charttime'], how='left')

final_df['charttime'] = pd.to_datetime(final_df['charttime'])
merged_df['admittime'] = pd.to_datetime(merged_df['admittime'])
merged_df['dischtime'] = pd.to_datetime(merged_df['dischtime'])

merged_data = pd.merge_asof(
    final_df.sort_values('charttime'),
    merged_df.sort_values('admittime'),
    left_on='charttime',
    right_on='admittime',
    by='subject_id',
    direction='backward',
    tolerance=pd.Timedelta('30D')  
)

merged_data = merged_data[
    (merged_data['charttime'] >= merged_data['admittime']) &
    (merged_data['charttime'] <= merged_data['dischtime'])
]

cols_to_check = [
    'albumin', 'c-reactive protein', 'chloride', 'creatinine', 'd-dimer', 'glucose',
    'hematocrit', 'hemoglobin', 'lactate', 'magnesium', 'ntprobnp', 'pco2', 'ph',
    'potassium', 'sodium', 'urea nitrogen'
]
high_missing_cols = ['c-reactive protein', 'd-dimer', 'ntprobnp']
merged_data.drop(columns=high_missing_cols, inplace=True)


imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=5, max_depth=5),
                           max_iter=5, random_state=42)

cols_to_impute = [
    'albumin', 'lactate', 'pco2', 'magnesium', 'ph', 'urea nitrogen', 'chloride',
    'creatinine', 'hemoglobin', 'sodium', 'hematocrit', 'potassium', 'glucose'
]

imputed_data = merged_data.copy()

imputed_values = imputer.fit_transform(imputed_data[cols_to_impute])
imputed_data[cols_to_impute] = imputed_values

merged_data=imputed_data

merged_data_1 = pd.get_dummies(merged_data, columns=['admission_type', 'discharge_location'])

insurance_risk = {
    'Medicare': 3,
    'Medicaid': 4,
    'Private': 1,
    'Self Pay': 2,
    'Government': 2
}

merged_data_1 = merged_data_1.drop('diagnosis', axis=1)

merged_data_1=merged_data_1.drop(['hadm_id','subject_id','charttime','hospital_expire_flag'],axis=1)

flag_columns = [
    'albumin_flag', 'creatinine_flag',
    'hematocrit_flag', 'hemoglobin_flag', 'magnesium_flag',
     'sodium_flag', 'urea nitrogen_flag','potassium_flag'
]

for col in flag_columns:
    if col in merged_data_1.columns:
        merged_data_1[col] = (
            merged_data_1[col]
            .astype(str)  
            .str.strip()  
            .replace({'abnormal': 1, 'delta': 2, 'nan': 0}) 
            .astype('int8')  
        )

merged_data_1['admittime'] = pd.to_datetime(merged_data_1['admittime'])
merged_data_1['dischtime'] = pd.to_datetime(merged_data_1['dischtime'])


merged_data_1['length_of_stay'] = (merged_data_1['dischtime'] - merged_data_1['admittime']).dt.days


merged_data_1['admit_weekday'] = merged_data_1['admittime'].dt.weekday


drop_cols = [
    'days_to_readmit', 
    'discharge_location_DEAD/EXPIRED',
    'admission_type_NEWBORN'
]
merged_data_1.drop(columns=drop_cols, inplace=True, errors='ignore')


if 'diagnosis' in merged_data_1.columns:
    merged_data_1= merged_data_1[merged_data_1['diagnosis'].str.contains('heart failure', case=False, na=False)]


selected_features = [
    
    'creatinine', 'creatinine_flag',
    'urea nitrogen', 'urea nitrogen_flag', 'sodium', 'sodium_flag',
    'potassium', 'potassium_flag', 'albumin', 'albumin_flag',
    'hemoglobin', 'hematocrit', 'hemoglobin_flag', 'hematocrit_flag',
    'magnesium', 'magnesium_flag',

   
    'admission_type_EMERGENCY', 'admission_type_URGENT',
    'discharge_location_HOME', 'discharge_location_HOME HEALTH CARE',
    'discharge_location_SNF', 'discharge_location_SHORT TERM HOSPITAL',
    'discharge_location_REHAB/DISTINCT PART HOSP', 'discharge_location_OTHER FACILITY',

    
    'insurance_risk',
    'length_of_stay', 'admit_weekday'
]

target_column = 'readmitted_30'


model_df = merged_data_1[selected_features + [target_column]].copy()

readmission_counts = model_df['readmitted_30'].value_counts()

class_0 = model_df[model_df['readmitted_30'] == 0]
class_1 = model_df[model_df['readmitted_30'] == 1]

class_0_sample = class_0.sample(n=20000, random_state=42)

combined = pd.concat([class_0_sample, class_1], axis=0)
X = combined.drop(columns=['readmitted_30'])
y = combined['readmitted_30']

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

joblib.dump(model, "model.pkl")
model = joblib.load("model.pkl")



