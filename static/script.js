/**
 * CreditWise — Frontend Interactions
 * Lightweight JavaScript enhancements for smooth UX, preset demos, and validation.
 */

document.addEventListener("DOMContentLoaded", () => {

  // -------------------------------------------------------------
  // 1. Result Page: Smooth Probability Bar Animation
  // -------------------------------------------------------------
  const probBar = document.getElementById("prob-bar");
  if (probBar) {
    const finalWidth = probBar.style.width;
    probBar.style.width = "0%";
    setTimeout(() => {
      probBar.style.width = finalWidth;
    }, 250);
  }

  // -------------------------------------------------------------
  // 2. Prediction Form: Quick Sample Presets (For Viva/Demo)
  // -------------------------------------------------------------
  function populateForm(sampleData) {
    for (const [key, value] of Object.entries(sampleData)) {
      const input = document.getElementById(key);
      if (input) {
        input.value = value;
        input.classList.remove("input-error");
      }
    }
  }

  const approvedBtn = document.getElementById("btn-sample-approved");
  if (approvedBtn) {
    approvedBtn.addEventListener("click", () => {
      populateForm({
        age: "38",
        gender: "Male",
        marital_status: "Married",
        dependents: "1",
        education_level: "Graduate",
        applicant_income: "55000",
        coapplicant_income: "35000",
        credit_score: "780",
        existing_loans: "0",
        dti_ratio: "0.18",
        savings: "500000",
        collateral_value: "1200000",
        loan_amount: "250000",
        loan_term: "24",
        loan_purpose: "Home",
        property_area: "Urban",
        employment_status: "Salaried",
        employer_category: "Government"
      });

      approvedBtn.innerHTML = "✓ Loaded Approved Profile";
      setTimeout(() => {
        approvedBtn.innerHTML = "⚡ Approved Sample";
      }, 1500);
    });
  }

  const rejectedBtn = document.getElementById("btn-sample-rejected");
  if (rejectedBtn) {
    rejectedBtn.addEventListener("click", () => {
      populateForm({
        age: "23",
        gender: "Female",
        marital_status: "Single",
        dependents: "3",
        education_level: "Not Graduate",
        applicant_income: "18000",
        coapplicant_income: "0",
        credit_score: "420",
        existing_loans: "3",
        dti_ratio: "0.75",
        savings: "10000",
        collateral_value: "50000",
        loan_amount: "800000",
        loan_term: "72",
        loan_purpose: "Personal",
        property_area: "Rural",
        employment_status: "Unemployed",
        employer_category: "Unemployed"
      });

      rejectedBtn.innerHTML = "✓ Loaded High-Risk Profile";
      setTimeout(() => {
        rejectedBtn.innerHTML = "⚠️ High-Risk Sample";
      }, 1500);
    });
  }

  // -------------------------------------------------------------
  // 3. Form Validation & Loading State
  // -------------------------------------------------------------
  const loanForm = document.getElementById("loan-form");
  const submitBtn = document.getElementById("submit-btn");

  if (loanForm && submitBtn) {
    const requiredInputs = loanForm.querySelectorAll("input[required], select[required]");

    // Clear error style on input change
    requiredInputs.forEach((field) => {
      field.addEventListener("input", () => {
        if (field.value.trim() !== "") {
          field.classList.remove("input-error");
        }
      });
      field.addEventListener("change", () => {
        if (field.value.trim() !== "") {
          field.classList.remove("input-error");
        }
      });
    });

    loanForm.addEventListener("submit", (e) => {
      let hasError = false;
      let firstInvalid = null;

      requiredInputs.forEach((field) => {
        if (!field.value || field.value.trim() === "") {
          field.classList.add("input-error");
          hasError = true;
          if (!firstInvalid) firstInvalid = field;
        } else {
          field.classList.remove("input-error");
        }
      });

      if (hasError) {
        e.preventDefault();
        if (firstInvalid) {
          firstInvalid.scrollIntoView({ behavior: "smooth", block: "center" });
          firstInvalid.focus();
        }
        return;
      }

      // Submit loading feedback
      submitBtn.disabled = true;
      submitBtn.style.opacity = "0.85";
      submitBtn.innerHTML = "<span>Predicting...</span>";
    });
  }
});
