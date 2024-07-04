import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


df = pd.read_csv('lung_cancer_data.csv')

comorbidity_columns = ['Comorbidity_Diabetes', 'Comorbidity_Hypertension', 'Comorbidity_Heart_Disease', 
                       'Comorbidity_Chronic_Lung_Disease', 'Comorbidity_Kidney_Disease', 
                       'Comorbidity_Autoimmune_Disease', 'Comorbidity_Other']

for column in comorbidity_columns:
    df[column] = df[column].map({'Yes': 1, 'No': 0})

df['Comorbidities_Count'] = df[comorbidity_columns].sum(axis=1)
df = df.drop(comorbidity_columns, axis=1)
print(df['Comorbidities_Count'])

categorical_columns = ['Gender', 'Smoking_History', 'Tumor_Location', 'Stage', 'Treatment', 
                       'Ethnicity', 'Insurance_Type', 'Family_History',
                    #    'Comorbidity_Diabetes', 'Comorbidity_Hypertension', 'Comorbidity_Heart_Disease', 
                    #     'Comorbidity_Chronic_Lung_Disease', 'Comorbidity_Kidney_Disease', 
                    #     'Comorbidity_Autoimmune_Disease', 'Comorbidity_Other'
                        ]
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

df = df.drop('Patient_ID', axis=1)

# corr_matrix = df.corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
#             linewidths=0.5, linecolor='black', fmt=".2f", annot_kws={"size": 10})
# plt.title('Correlation Matrix')
# plt.show()

weird_columns = [
    "Hemoglobin_Level",	
    "White_Blood_Cell_Count",	
    "Platelet_Count",	
    # "Albumin_Level",	
    # "Alkaline_Phosphatase_Level",	
    # "Alanine_Aminotransferase_Level",	
    "Aspartate_Aminotransferase_Level",	
    "Creatinine_Level",	
    "LDH_Level",	
    "Calcium_Level",	
    # "Phosphorus_Level",	
    # "Glucose_Level",	
    # "Potassium_Level",	
    # "Sodium_Level" 
    ]
df=df.drop(weird_columns, axis=1)
other_columns = [
	# "Age",
    # "Gender",
	# "Smoking_History",
    # "Tumor_Size_mm",
    # "Tumor_Location",
    # "Stage",
    # "Treatment",
    # "Survival_Months",
    # "Ethnicity",
    "Insurance_Type",
    # "Family_History",
    # "Performance_Status",	
    # "Blood_Pressure_Systolic",	
    # "Blood_Pressure_Diastolic",
    # "Blood_Pressure_Pulse"

]
df=df.drop(other_columns, axis=1)

X = df.drop('Stage', axis=1)
imputer = SimpleImputer(strategy="mean")
imputer.fit(X)
X = imputer.transform(X)
y = df['Stage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(np.shape(X_train))  

classifier = SVC(kernel='rbf', gamma='auto', C=5, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))

macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"Macro F1 Score: {macro_f1}")

cm = confusion_matrix(y_test, y_pred)
class_labels = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")