import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np


# Load dataset from CSV file
df = pd.read_csv('pds.csv')

# Drop rows with missing Domain (which means no placement)
df = df.dropna(subset=['Domain'])

# Remove companies with only one sample (optional, but recommended for stratification)
company_counts = df['Company'].value_counts()
valid_companies = company_counts[company_counts > 1].index
df = df[df['Company'].isin(valid_companies)]

# Prepare features using CGPA and Domain
X = df[['CGPA', 'Domain']].copy()

# Load the previously saved domain encoder and transform Domain
with open('domain_encoder.pkl', 'rb') as f:
    le_domain = pickle.load(f)
X['Domain_encoded'] = le_domain.transform(X['Domain'])
X = X[['CGPA', 'Domain_encoded']]

# Prepare target labels using the company encoder
y = df['Company']
with open('company_encoder.pkl', 'rb') as f:
    le_company = pickle.load(f)
y_encoded = le_company.transform(y)

# Split dataset into training and testing (using stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)



# Load the trained XGBoost model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Generate predictions on the test set
y_pred = model.predict(X_test)

unique_labels = np.sort(np.unique(y_test))
# Use these indices to extract the corresponding target names from the encoder
target_names = le_company.classes_[unique_labels]

# Generate the classification report using the specified labels and target names
report = classification_report


# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
# Calculate F1 score using macro averaging (suitable for multi-class)
f1 = f1_score(y_test, y_pred, average='macro')
report = classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the results
print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score (Macro): {f1:.4f}\n")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)
