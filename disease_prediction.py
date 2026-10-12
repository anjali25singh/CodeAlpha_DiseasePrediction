import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

# 1. Load data
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
print("Shape:", X.shape)
print("Missing values:", X.isnull().sum().sum())
print("Class distribution:\n", y.value_counts())

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scaling (SVM & Logistic Regression)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4. Models
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), True),
    "SVM": (SVC(probability=True, kernel="rbf"), True),
    "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=42), False),
    "XGBoost": (XGBClassifier(n_estimators=200, learning_rate=0.1,
                              eval_metric="logloss", random_state=42), False),
}

results = []
trained = {}
for name, (model, needs_scaling) in models.items():
    Xtr = X_train_s if needs_scaling else X_train
    Xte = X_test_s if needs_scaling else X_test
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    trained[name] = (model, Xte, pred)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    })

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print("\n", results_df.round(4).to_string(index=False))

# 5. Best model ka confusion matrix + report
best = results_df.iloc[0]["Model"]
_, _, best_pred = trained[best]
print(f"\nBest model: {best}")
print(classification_report(y_test, best_pred, target_names=data.target_names))

cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.title(f"Confusion Matrix - {best}")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# 6. Feature importance (Random Forest)
rf = trained["Random Forest"][0]
imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
imp.sort_values().plot(kind="barh", title="Top 10 Features (Random Forest)")
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()