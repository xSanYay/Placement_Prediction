import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset from CSV file
df = pd.read_csv('pds.csv')

# Drop rows with missing Domain (which means no placement)
df = df.dropna(subset=['Domain'])

# Remove companies with only one sample (optional, but recommended for stratification)
company_counts = df['Company'].value_counts()
valid_companies = company_counts[company_counts > 1].index
df = df[df['Company'].isin(valid_companies)]

# Use CGPA and Domain as features
X = df[['CGPA', 'Domain']].copy()

# Encode the Domain column
le_domain = LabelEncoder()
X['Domain_encoded'] = le_domain.fit_transform(X['Domain'])
X = X[['CGPA', 'Domain_encoded']]

# The target is the Company column
y = df['Company']
le_company = LabelEncoder()
y_encoded = le_company.fit_transform(y)

# Print class distribution to understand the data
print("Class distribution:", pd.Series(y_encoded).value_counts().to_dict())

# Increase test_size to ensure test set is large enough for all classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Train the XGBoost classifier
model = xgb.XGBClassifier(num_class=len(le_company.classes_), eval_metric='mlogloss')
model.fit(X_train, y_train)

# Save the model and encoders to disk
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('domain_encoder.pkl', 'wb') as f:
    pickle.dump(le_domain, f)
with open('company_encoder.pkl', 'wb') as f:
    pickle.dump(le_company, f)

# Create a mapping of Company to average CTC for filtering purposes
company_ctc = df.groupby('Company')['CTC'].mean().to_dict()
with open('company_ctc.pkl', 'wb') as f:
    pickle.dump(company_ctc, f)

print("Training completed and files saved.")
