import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('lung_cancer_data.csv')

categorical_columns = ['Gender','Smoking_History','Tumor_Location','Stage','Treatment','Ethnicity','Insurance_Type','Family_History','Comorbidity_Diabetes','Comorbidity_Hypertension','Comorbidity_Heart_Disease','Comorbidity_Chronic_Lung_Disease','Comorbidity_Kidney_Disease','Comorbidity_Autoimmune_Disease','Comorbidity_Other']
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

df = df.drop('Patient_ID', axis=1)

X = df.drop('Stage', axis=1)
y = df['Stage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
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
plt.show()