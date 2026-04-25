# =========================================
# LIGAND–RECEPTOR BINDING PREDICTION PROJECT
# =========================================

# Import libraries
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# =========================================
# STEP 1: Load SDF File
# =========================================

file_path = "C:/Users/hsgee/Downloads/monomers_12852_2D.sdf"

supplier = Chem.SDMolSupplier(file_path)
molecules = [mol for mol in supplier if mol is not None]

print("Total molecules:", len(molecules))

# =========================================
# STEP 2: Extract Molecular Features
# =========================================

data = []

for mol in molecules:
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hdonor = Descriptors.NumHDonors(mol)
        hacceptor = Descriptors.NumHAcceptors(mol)

        data.append([mw, logp, hdonor, hacceptor])
    except:
        continue

df = pd.DataFrame(data, columns=["MolWt", "LogP", "HDonor", "HAcceptor"])

print("\nSample Data:")
print(df.head())

# =========================================
# STEP 3: Create Labels (Demo purpose)
# =========================================

# Using median as threshold (better than fixed 300)
threshold = df["MolWt"].median()
df["Binding"] = df["MolWt"].apply(lambda x: 1 if x > threshold else 0)

# =========================================
# STEP 4: Prepare Data
# =========================================

X = df.drop("Binding", axis=1)
y = df["Binding"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# STEP 5: Train Model
# =========================================

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================================
# STEP 6: Evaluate Model
# =========================================

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC-AUC Score
y_prob = model.predict_proba(X_test)[:, 1]
print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))

# =========================================
# STEP 7: Predict New Molecule
# =========================================

def predict_molecule(mol):
    features = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol)
    ]
    
    features_df = pd.DataFrame([features], columns=X.columns)
    
    pred = model.predict(features_df)
    
    return "Binds" if pred[0] == 1 else "Does Not Bind"

# Test with first molecule
test_mol = molecules[0]
print("\nPrediction for sample molecule:", predict_molecule(test_mol))

# =========================================
# END OF PROJECT
# =========================================
