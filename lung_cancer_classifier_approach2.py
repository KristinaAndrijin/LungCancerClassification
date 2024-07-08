import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('lung_cancer_data.csv')

categorical_columns = ['Gender', 'Smoking_History', 'Tumor_Location', 'Stage', 'Treatment',
                       'Ethnicity', 'Insurance_Type', 'Family_History',
                       'Comorbidity_Diabetes', 'Comorbidity_Hypertension', 'Comorbidity_Heart_Disease',
                       'Comorbidity_Chronic_Lung_Disease', 'Comorbidity_Kidney_Disease',
                       'Comorbidity_Autoimmune_Disease', 'Comorbidity_Other'
                       ]
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

df = df.drop('Patient_ID', axis=1)

X = df.drop('Stage', axis=1)
y = df['Stage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_dt_params = {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 5}
best_knn_params = {'n_neighbors': 10, 'weights': 'distance'}
best_svm_params = {'C': 5, 'gamma': 'scale', 'kernel': 'rbf'}

clf1 = DecisionTreeClassifier(**best_dt_params)
clf2 = KNeighborsClassifier(**best_knn_params)
clf3 = SVC(**best_svm_params, probability=True)

base_estimators = [('dt', clf1), ('knn', clf2), ('svm', clf3)]

voting_clf = VotingClassifier(estimators=base_estimators, voting='soft')

voting_clf.fit(X_train_scaled, y_train)

y_pred = voting_clf.predict(X_test_scaled)

print("Voting Classifier with Grid Search Parameters:")
print(classification_report(y_test, y_pred))
macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"Macro F1 Score: {macro_f1}")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders['Stage'].classes_, yticklabels=label_encoders['Stage'].classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
