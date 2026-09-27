"""
app.py  —  CreditWise: Loan Approval Prediction System
======================================================
College Minor Project: Supervised Machine Learning
Backend: Flask + scikit-learn (Gaussian Naïve Bayes)

Data Flow:
    1. User enters applicant & loan data into HTML form
    2. Flask receives POST request with form payload
    3. Input is preprocessed using the exact pipeline from MinorProject.ipynb:
       - LabelEncoder for Education_Level (Graduate -> 0, Not Graduate -> 1)
       - OneHotEncoder for 6 categorical features (with drop='first')
       - Concatenation in exact 27-feature training order
       - StandardScaler fitted during model training
    4. GaussianNB model predicts:
       - Class 1 = Loan Approved
       - Class 0 = Loan Rejected
       - predict_proba() computes confidence percentages
    5. Result page displays verdict, probability meter, and application summary.

Routes:
    GET  /          -> Landing page
    GET  /predict   -> Loan application form
    POST /predict   -> Process form, predict, render result page
"""

import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

# ------------------------------------------------------------------
# 1. Flask Application Setup
# ------------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------------
# 2. Load Model Artifacts ONCE at Startup
#    (Loaded once in memory; never retrained on user requests)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "loan_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Please run 'python save_model.py' to generate the trained model."
    )

with open(MODEL_PATH, "rb") as f:
    artifacts = pickle.load(f)

model           = artifacts["model"]           # GaussianNB trained model
scaler          = artifacts["scaler"]          # StandardScaler fitted on X_train
ohe             = artifacts["ohe"]             # OneHotEncoder fitted on categorical cols
le_education    = artifacts["le_education"]    # LabelEncoder for Education_Level
le_target       = artifacts["le_target"]       # LabelEncoder for Loan_Approved (0=No, 1=Yes)
feature_columns = artifacts["feature_columns"] # Exact 27-column feature order
ohe_cols        = artifacts["ohe_cols"]        # 6 one-hot encoded column names

print(f"[CreditWise] Model artifacts successfully loaded from: {MODEL_PATH}")
print(f"[CreditWise] Total model features: {len(feature_columns)}")


# ------------------------------------------------------------------
# 3. Preprocessing Helper Function
#    (Mirrors MinorProject.ipynb cells 16-26)
# ------------------------------------------------------------------
def preprocess_input(form_data):
    """
    Transforms raw user form data into a scaled feature array
    matching the exact feature order and distribution of the training data.

    Args:
        form_data (dict): Mapping of input names to string values from HTML form.

    Returns:
        np.ndarray: Scaled array of shape (1, 27) ready for model.predict().
    """
    # 3.1 Validate & extract numerical inputs
    try:
        applicant_income    = float(form_data["applicant_income"])
        coapplicant_income  = float(form_data["coapplicant_income"])
        age                 = float(form_data["age"])
        dependents          = float(form_data["dependents"])
        credit_score        = float(form_data["credit_score"])
        existing_loans      = float(form_data["existing_loans"])
        dti_ratio           = float(form_data["dti_ratio"])
        savings             = float(form_data["savings"])
        collateral_value    = float(form_data["collateral_value"])
        loan_amount         = float(form_data["loan_amount"])
        loan_term           = float(form_data["loan_term"])
    except (ValueError, KeyError) as e:
        raise ValueError(f"Invalid numeric input: {e}")

    # 3.2 Label Encode Education_Level (Graduate -> 0, Not Graduate -> 1)
    education_raw = form_data["education_level"]
    education_encoded = int(le_education.transform([education_raw])[0])

    # 3.3 One-Hot Encode the 6 categorical features
    cat_df = pd.DataFrame([[
        form_data["employment_status"],
        form_data["marital_status"],
        form_data["gender"],
        form_data["loan_purpose"],
        form_data["property_area"],
        form_data["employer_category"],
    ]], columns=ohe_cols)

    ohe_transformed = ohe.transform(cat_df)  # Shape (1, 15)

    # 3.4 Build numeric feature vector in exact order:
    # [Applicant_Income, Coapplicant_Income, Age, Dependents, Credit_Score,
    #  Existing_Loans, DTI_Ratio, Savings, Collateral_Value, Loan_Amount,
    #  Loan_Term, Education_Level]
    numeric_values = np.array([[
        applicant_income,
        coapplicant_income,
        age,
        dependents,
        credit_score,
        existing_loans,
        dti_ratio,
        savings,
        collateral_value,
        loan_amount,
        loan_term,
        education_encoded,
    ]])  # Shape (1, 12)

    # 3.5 Combine numeric (12) + OHE (15) into 27 features
    X_raw = np.hstack([numeric_values, ohe_transformed])

    # Convert to DataFrame with feature names to preserve alignment with StandardScaler
    X_df = pd.DataFrame(X_raw, columns=feature_columns)

    # 3.6 Standardize features using the saved StandardScaler
    X_scaled = scaler.transform(X_df)

    return X_scaled


# ------------------------------------------------------------------
# 4. Route Handlers
# ------------------------------------------------------------------

@app.route("/")
def home():
    """Renders the landing page."""
    return render_template("index.html")


@app.route("/predict", methods=["GET"])
def predict_form():
    """Renders the loan application prediction form."""
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles form submission:
    - Validates inputs
    - Executes preprocessing pipeline
    - Generates prediction & class probabilities
    - Renders the result page
    """
    errors = []

    # 4.1 Check presence of all 18 required features
    required_fields = [
        "applicant_income", "coapplicant_income", "age", "dependents",
        "credit_score", "existing_loans", "dti_ratio", "savings",
        "collateral_value", "loan_amount", "loan_term",
        "education_level", "employment_status", "marital_status",
        "gender", "loan_purpose", "property_area", "employer_category",
    ]
    for field in required_fields:
        val = request.form.get(field, "").strip()
        if not val:
            field_label = field.replace("_", " ").title()
            errors.append(f"Please provide a valid value for {field_label}.")

    if errors:
        return render_template("predict.html", errors=errors, form_data=request.form)

    # 4.2 Sensible range and boundary validation
    try:
        age = float(request.form["age"])
        if not (18 <= age <= 80):
            errors.append("Age must be between 18 and 80.")

        credit_score = float(request.form["credit_score"])
        if not (300 <= credit_score <= 900):
            errors.append("Credit Score must be between 300 and 900.")

        dti_ratio = float(request.form["dti_ratio"])
        if not (0.0 <= dti_ratio <= 1.0):
            errors.append("DTI Ratio must be between 0.0 and 1.0.")

        applicant_income = float(request.form["applicant_income"])
        if applicant_income <= 0:
            errors.append("Applicant Income must be greater than 0.")

        loan_amount = float(request.form["loan_amount"])
        if loan_amount <= 0:
            errors.append("Loan Amount must be greater than 0.")

    except ValueError:
        errors.append("Please enter valid numerical values for financial inputs.")

    if errors:
        return render_template("predict.html", errors=errors, form_data=request.form)

    # 4.3 Preprocess and Predict
    try:
        X_scaled = preprocess_input(request.form)

        # Binary prediction (0 = Rejected, 1 = Approved)
        prediction = int(model.predict(X_scaled)[0])

        # Prediction probabilities [P(Rejection), P(Approval)]
        probabilities = model.predict_proba(X_scaled)[0]
        prob_approved = round(float(probabilities[1]) * 100, 1)
        prob_rejected = round(float(probabilities[0]) * 100, 1)

        is_approved = (prediction == 1)
        result_label = "Approved" if is_approved else "Rejected"

        # DEBUG — confirm values before rendering
        print(f"[DEBUG] prediction={prediction}, is_approved={is_approved}")
        print(f"[DEBUG] prob_approved={prob_approved}, prob_rejected={prob_rejected}")

    except Exception as e:
        return render_template(
            "predict.html",
            errors=[f"Prediction processing error: {str(e)}"],
            form_data=request.form,
        )

    # 4.4 Render the Result Card Page
    return render_template(
        "result.html",
        result_label=result_label,
        is_approved=is_approved,
        prob_approved=prob_approved,
        prob_rejected=prob_rejected,
        form_data=request.form,
    )


# ------------------------------------------------------------------
# 5. Application Entry Point
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("[CreditWise] Starting Flask development server...")
    print("[CreditWise] Accessible at http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=True)
