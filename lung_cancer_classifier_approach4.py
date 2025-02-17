import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

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
b = X_train

imputer = KNNImputer(n_neighbors=100)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler(with_mean=True, with_std=True)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components='mle', svd_solver='full')
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

clf = HistGradientBoostingClassifier(learning_rate=0.2, max_depth=20, max_leaf_nodes=63, min_samples_leaf=50,
                                     random_state=80)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Classification report HistGradientBoostingClassifier")
print(classification_report(y_test, y_pred))

macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')
print(f"Macro F1 Score HistGradientBoostingClassifier: {macro_f1}")
print(f"Micro F1 Score HistGradientBoostingClassifier: {micro_f1}")
cm = confusion_matrix(y_test, y_pred)
class_labels = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print(30 * '=')

dt = DecisionTreeClassifier(max_depth=50)
clf = AdaBoostClassifier(learning_rate=0.1, estimator=dt, random_state=80)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Classification report AdaBoostClassifier")
print(classification_report(y_test, y_pred))

macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')
print(f"Macro F1 Score AdaBoostClassifier: {macro_f1}")
print(f"Micro F1 Score AdaBoostClassifier: {micro_f1}")
cm = confusion_matrix(y_test, y_pred)
class_labels = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print(30 * '=')

clf = ExtraTreesClassifier(bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=100,
                           random_state=80)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Classification report ExtraTreesClassifier")
print(classification_report(y_test, y_pred))

macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')
print(f"Macro F1 Score ExtraTreesClassifier: {macro_f1}")
print(f"Micro F1 Score ExtraTreesClassifier: {micro_f1}")
cm = confusion_matrix(y_test, y_pred)
class_labels = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print(30 * '=')
