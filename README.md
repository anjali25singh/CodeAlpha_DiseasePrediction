# Disease Prediction from Medical Data

Predicts whether a breast tumor is **malignant or benign** using machine learning classification on structured medical data.

## Objective
Predict the possibility of disease based on patient data using classification techniques.

## Dataset
Breast Cancer Wisconsin (Diagnostic) dataset, loaded directly from `scikit-learn`.
- 569 samples, 30 numerical features (radius, texture, perimeter, area, smoothness, etc.)
- No missing values
- Target: 0 = malignant (212 samples), 1 = benign (357 samples)

## Approach
1. Load data and check missing values and class distribution
2. Stratified train-test split (80/20), 455 train and 114 test samples
3. Feature scaling with `StandardScaler` (for Logistic Regression and SVM)
4. Train 4 models: Logistic Regression, SVM, Random Forest, XGBoost
5. Evaluate using Accuracy, Precision, Recall, F1-Score and ROC-AUC
6. Plot confusion matrix and feature importance

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 |
| SVM | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9950 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9931 |
| XGBoost | 0.9474 | 0.9459 | 0.9722 | 0.9589 | 0.9937 |

**Best model:** Logistic Regression (Accuracy 98.25%, ROC-AUC 0.9954)

### Classification Report (Logistic Regression)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Malignant | 0.98 | 0.98 | 0.98 | 42 |
| Benign | 0.99 | 0.99 | 0.99 | 72 |
| **Accuracy** | | | **0.98** | 114 |

