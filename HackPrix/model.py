import pandas as pd
import numpy as np
import os
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

# Define paths to your local dataset files
base_dir = r'C:\Users\rohit\Desktop\hackprix'
dataset_path = os.path.join(base_dir, 'dataset.csv')
symptom_severity_path = os.path.join(base_dir, 'Symptom-severity.csv')
symptom_description_path = os.path.join(base_dir, 'symptom_Description.csv')
symptom_precaution_path = os.path.join(base_dir, 'symptom_precaution.csv')
model_save_path = os.path.join(base_dir, 'model')

# Create model directory if it doesn't exist
os.makedirs(model_save_path, exist_ok=True)

# Load dataset
df = pd.read_csv(dataset_path)

# Shuffle the DataFrame
df = shuffle(df, random_state=42)

# Clean column names
df.columns = df.columns.str.replace('_', ' ').str.strip()
df = df.fillna(0)

# Load symptom severity
df1 = pd.read_csv(symptom_severity_path)
symptom_list = df1['Symptom'].tolist()

dfx = pd.DataFrame()
dfx["Disease"] = df["Disease"]
dfx[symptom_list] = 0
for index, row in df.iterrows():
    for symptom in df.columns[1:]:
        if row[symptom] != 0:
            dfx.loc[index, row[symptom]] = 1

dfx = dfx.fillna(0)
dfx[dfx.columns[1:]] = dfx[dfx.columns[1:]].astype('int')
dfx.columns = dfx.columns.str.strip()

# Drop unwanted columns only if they exist in the DataFrame
columns_to_drop = ['foul smell of urine', 'dischromic patches', 'spotting urination']
existing_columns = [col for col in columns_to_drop if col in dfx.columns]
dfx = dfx.drop(columns=existing_columns)

data = dfx.iloc[:, 1:].values
labels = dfx['Disease'].values

x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=0.7, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val = le.transform(y_val)
y_classes = le.classes_

# Define classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier(verbose=-1),
    'CatBoost': CatBoostClassifier(silent=True),
    'GradientBoost': GradientBoostingClassifier(),
    'ExtraTrees': ExtraTreesClassifier()
}

# K-fold Cross Validation model evaluation
kfold = KFold(n_splits=10, shuffle=True, random_state=1)

for name, clf in classifiers.items():
    cv_scores = cross_val_score(clf, x_train, y_train, cv=kfold, scoring='f1_weighted')
    print(f'{name} cross-validation mean F1 score: %.3f' % cv_scores.mean())

    # Train and test each classifier
    clf.fit(x_train, y_train)

    test_predictions = clf.predict(x_test)
    test_f1 = f1_score(y_test, test_predictions, average='weighted')
    test_roc = roc_auc_score(y_test, clf.predict_proba(x_test), multi_class='ovr')
    print(f'{name} test F1 Score: {test_f1:.4f}, AUC-ROC Score: {test_roc:.4f}')

    val_predictions = clf.predict(x_val)
    val_f1 = f1_score(y_val, val_predictions, average='weighted')
    val_roc = roc_auc_score(y_val, clf.predict_proba(x_val), multi_class='ovr')
    print(f'{name} validation F1 Score: {val_f1:.4f}, AUC-ROC Score: {val_roc:.4f}')
    
    # Save the trained model
    with open(os.path.join(model_save_path, f"{name}.pkl"), "wb") as model_file:
        pickle.dump(clf, model_file)

# Load description and precaution files
desc = pd.read_csv(symptom_description_path)
prec = pd.read_csv(symptom_precaution_path)

# Prediction function
def predd(m, X):
    proba = m.predict_proba(X)
    top5_idx = np.argsort(proba[0])[-5:][::-1]
    top5_proba = np.sort(proba[0])[-5:][::-1]
    top5_diseases = y_classes[top5_idx]

    for i in range(5):
        disease = top5_diseases[i]
        probability = top5_proba[i]
        print("Disease Name: ", disease)
        print("Probability: ", probability)
        if disease in desc["Disease"].unique():
            disp = desc[desc['Disease'] == disease].values[0][1]
            print("Disease Description: ", disp)

        if disease in prec["Disease"].unique():
            c = np.where(prec['Disease'] == disease)[0][0]
            precaution_list = [prec.iloc[c, j] for j in range(1, len(prec.iloc[c]))]
            print("Recommended Things to do at home: ")
            for precaution in precaution_list:
                print(precaution)
        print("\n")

# User input
symptom_input = input("Enter symptoms separated by commas: ").split(',')
symptom_input = [symptom.strip() for symptom in symptom_input]

# Create symptom input Series
symptom_series = pd.Series([0] * len(dfx.columns[1:]), index=dfx.columns[1:])
for symptom in symptom_input:
    if symptom in symptom_series.index:
        symptom_series[symptom] = 1

# Drop the unwanted columns from symptom input
symptom_series = symptom_series.drop(columns_to_drop, errors='ignore')

# Load the trained model
with open(os.path.join(model_save_path, "ExtraTrees.pkl"), 'rb') as f:
    m = pickle.load(f)

# Ensure the input features match the model's expected features
t = symptom_series.to_numpy().reshape(1, -1)
print("Input shape: ", t.shape)
print("Model expects: ", m.n_features_in_)
predd(m, t)
