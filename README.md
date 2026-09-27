# CreditWise – Loan Approval Prediction System

**Live Demo:** https://creditwise-loansystem.onrender.com

---

## Overview

CreditWise is a Flask-based web application that uses a trained Machine Learning model to predict whether a loan application is likely to be **Approved** or **Rejected**. It was developed as a college Minor Project for the Supervised Machine Learning course.

---

## Problem Statement

Banks and financial institutions process thousands of loan applications manually, which is slow, inconsistent, and prone to human bias. There is a need for a data-driven, automated system that can provide a quick preliminary assessment of loan eligibility based on applicant and financial information.

---

## Objective

To build an end-to-end ML-powered web application that:
- Accepts loan applicant information through a web form
- Preprocesses the input using the same pipeline as the training data
- Uses a trained Gaussian Naive Bayes classifier to predict loan approval
- Displays the result along with calibrated approval/rejection probabilities

---

## Machine Learning Model

| Property | Detail |
|---|---|
| **Algorithm** | Gaussian Naive Bayes (`GaussianNB`) |
| **Probability Calibration** | `CalibratedClassifierCV` with sigmoid method |
| **Test Accuracy** | 87% |
| **Dataset Size** | 1,000 applicant records |
| **Features Used** | 18 input features → 27 after encoding |

**Why Gaussian Naive Bayes?**  
GaussianNB was selected as the best-performing model after comparing multiple classifiers in the notebook. It is fast, interpretable, and works well with mixed feature types.

**Why Probability Calibration?**  
GaussianNB is known for producing overconfident probabilities (0% or 100%) due to numerical underflow when multiplying many Gaussian likelihoods. `CalibratedClassifierCV` corrects this to produce realistic probability scores.

---

## Features Used

### Numerical Features (12)
| Feature | Description |
|---|---|
| Applicant Income | Monthly income of the primary applicant |
| Co-applicant Income | Monthly income of the co-applicant |
| Age | Age of the applicant |
| Dependents | Number of financial dependents |
| Credit Score | CIBIL/Credit score (300–900) |
| Existing Loans | Number of existing active loans |
| DTI Ratio | Debt-to-Income ratio (0.0–1.0) |
| Savings | Total savings amount |
| Collateral Value | Value of asset offered as collateral |
| Loan Amount | Requested loan amount |
| Loan Term | Loan repayment term in months |
| Education Level | Graduate (0) / Not Graduate (1) — Label Encoded |

### Categorical Features (6, One-Hot Encoded with `drop='first'`)
| Feature | Categories |
|---|---|
| Employment Status | Salaried, Self-employed, Contract, Unemployed |
| Marital Status | Married, Single |
| Gender | Male, Female |
| Loan Purpose | Home, Car, Education, Business, Personal |
| Property Area | Urban, Semiurban, Rural |
| Employer Category | Government, Private, MNC, Business, Unemployed |

---

## Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, CSS3, Vanilla JavaScript, Jinja2 Templates |
| **Backend** | Python 3.13, Flask 3.x |
| **ML Library** | scikit-learn 1.7 |
| **Data Processing** | pandas, NumPy |
| **Model Serialization** | pickle |
| **Production Server** | Gunicorn |
| **Deployment** | Render |
| **Version Control** | Git + GitHub |

---

## Project Structure

```
CreditWise/
│
├── app.py                  ← Flask backend (routes + prediction logic)
├── Procfile                ← Gunicorn startup command for Render
├── requirements.txt        ← Python dependencies
├── .gitignore              ← Files excluded from Git
├── README.md               ← This file
│
├── model/
│   └── loan_model.pkl      ← Trained model + preprocessing artifacts
│
├── templates/
│   ├── index.html          ← Home / landing page
│   ├── predict.html        ← Loan application form
│   └── result.html         ← Prediction result page
│
└── static/
    ├── style.css           ← Fintech-themed CSS design system
    └── script.js           ← JavaScript (form presets, animations)
```

---

## How It Works

```
User fills form (18 fields)
        ↓
Flask receives POST request
        ↓
Preprocessing pipeline:
  1. Extract numeric fields (11 values)
  2. Label Encode Education_Level (Graduate→0, Not Graduate→1)
  3. One-Hot Encode 6 categorical columns (drop='first')
  4. Concatenate into 27-feature vector
  5. Standardize using fitted StandardScaler
        ↓
CalibratedGaussianNB predicts:
  → Class 1 = Approved | Class 0 = Rejected
  → predict_proba() → calibrated confidence %
        ↓
Result page shows:
  → Verdict (Approved / Rejected)
  → Approval Probability %
  → Rejection Probability %
  → Visual probability bar
  → Application summary
```

---

## How to Run Locally

### Prerequisites
- Python 3.10+
- pip

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/Madhura-Kedar/CreditWise-LoanSystem.git
cd CreditWise-LoanSystem
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Flask app**
```bash
python app.py
```

**5. Open in browser**
```
http://127.0.0.1:5000/
```

> **Note:** The trained model (`model/loan_model.pkl`) is included in the repository. You do not need to retrain the model to run the app.

---

## Deployment

This application is deployed on **Render** using Gunicorn as the production WSGI server.

**Start command used by Render:**
```
gunicorn app:app
```

**Live URL:** *(Will be added after Render deployment)*

---

## Limitations

- The model was trained on a synthetic/sample dataset of 1,000 records — predictions should not be treated as real financial decisions.
- GaussianNB assumes feature independence, which may not hold for all loan features.
- The application does not store user data or prediction history.
- Probability scores are calibrated estimates and may not reflect true statistical likelihood.

---

## Future Scope

- Train on a larger, real-world loan dataset for improved accuracy
- Add more ML models (Random Forest, XGBoost) and compare performance
- Implement user authentication and prediction history
- Add SHAP or LIME explainability to show which features influenced the decision
- Build an admin dashboard to monitor prediction statistics

---

## Academic Disclaimer

> This application was built as a college Minor Project for educational purposes.  
> Predictions generated by this system are **not** real financial advice and should **not** be used for actual loan decisions.

---

*Developed by Madhura Kedar | SML Minor Project | 2026*
