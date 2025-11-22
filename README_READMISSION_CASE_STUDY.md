# Hospital Readmission Risk Prediction - Case Study

This repository contains a comprehensive case study application for predicting patient readmission risk within 30 days of discharge.

## Files

- **`hospital_readmission_case_study.md`**: Complete case study documentation covering all required sections
- **`hospital_readmission_model.py`**: Python implementation with model training, evaluation, and visualization
- **`requirements_readmission.txt`**: Python dependencies

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_readmission.txt
```

### 2. Run the Model Demonstration

```bash
python hospital_readmission_model.py
```

This will:
- Generate hypothetical patient data
- Train an XGBoost model with regularization
- Evaluate model performance
- Generate confusion matrix and performance metrics
- Create visualizations (confusion matrix, ROC curve, feature importance)

### 3. View Results

The script generates three visualization files:
- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curve.png` - ROC curve with AUC score
- `feature_importance.png` - Top features contributing to predictions

## Case Study Sections

The case study document (`hospital_readmission_case_study.md`) covers:

1. **Problem Scope** (5 points)
   - Problem definition
   - Objectives
   - Stakeholders

2. **Data Strategy** (10 points)
   - Proposed data sources (EHRs, demographics, etc.)
   - 2 ethical concerns (patient privacy, algorithmic bias)
   - Preprocessing pipeline design

3. **Model Development** (10 points)
   - Model selection and justification (XGBoost)
   - Confusion matrix with hypothetical data
   - Precision, recall, and other performance metrics

4. **Deployment** (10 points)
   - Integration steps into hospital system
   - HIPAA compliance measures

5. **Optimization** (5 points)
   - Method to address overfitting (regularization with cross-validation)

## Model Performance

The model demonstrates:
- **Accuracy**: ~90%
- **Precision**: ~75%
- **Recall**: ~66.7%
- **AUC-ROC**: Varies based on data

## Key Features

- **Regularization**: Prevents overfitting using early stopping, cross-validation, and regularization parameters
- **HIPAA Compliance**: Comprehensive security and privacy measures
- **Ethical Considerations**: Addresses bias and fairness in healthcare AI
- **Production-Ready**: Includes deployment strategy and integration steps

## Notes

- The data used is **hypothetical** and generated for demonstration purposes
- In a real implementation, actual patient data would be used with proper HIPAA safeguards
- The model parameters can be tuned based on actual data characteristics

