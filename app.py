from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model, encoders, and company CTC mapping
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('domain_encoder.pkl', 'rb') as f:
    le_domain = pickle.load(f)
with open('company_encoder.pkl', 'rb') as f:
    le_company = pickle.load(f)
with open('company_ctc.pkl', 'rb') as f:
    company_ctc = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        cgpa = float(data.get('cgpa'))
        domain_input = data.get('domain')
        expected_salary = float(data.get('expected_salary'))
    except Exception as e:
        return jsonify({'error': 'Invalid input. Please provide cgpa (float), domain (string), and expected_salary (float).'}), 400
    
    # Process the domain: assume the input directly matches one of the known domains.
    try:
        domain_encoded = le_domain.transform([domain_input])[0]
    except Exception as e:
        return jsonify({'error': 'Domain not recognized. Please provide a valid domain.'}), 400
    
    # Prepare the feature vector for prediction
    X_input = np.array([[cgpa, domain_encoded]])
    
    # Get prediction probabilities for each company
    probs = model.predict_proba(X_input)[0]
    
    # Sort companies by probability (highest first)
    sorted_indices = np.argsort(probs)[::-1]
    
    recommended_companies = []
    for idx in sorted_indices:
        company_name = le_company.inverse_transform([idx])[0]
        # Retrieve the average CTC for the company
        company_salary = company_ctc.get(company_name, 0)
        # Filter based on candidate's expected salary
        if company_salary >= expected_salary:
            recommended_companies.append({
                'company': company_name,
                'predicted_probability': float(probs[idx]),
                'average_CTC': company_salary
            })
    
    if not recommended_companies:
        return jsonify({'message': 'No companies available matching your expected salary.'})
    
    return jsonify({'recommended_companies': recommended_companies})

if __name__ == '__main__':
    app.run(debug=True,port=4567)


