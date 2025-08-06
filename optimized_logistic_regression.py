import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
features = data.feature_names

# Simulated grouping (for demo purposes)
# G1: 0-9, G2: 10-19, G3: 20-29
G1_idx = list(range(0, 10))
G2_idx = list(range(10, 20))
G3_idx = list(range(20, 30))

X_G1, X_G2, X_G3 = X[:, G1_idx], X[:, G2_idx], X[:, G3_idx]

# Apply adaptive scaling
X_G1_scaled = StandardScaler().fit_transform(X_G1)
X_G2_scaled = MinMaxScaler().fit_transform(X_G2)
X_G3_scaled = RobustScaler().fit_transform(X_G3)

# Concatenate scaled groups
X_scaled = np.concatenate([X_G1_scaled, X_G2_scaled, X_G3_scaled], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Group-specific regularization
# G1+G2: L2 Regularization
# G3: L1 Regularization

# Split train and test into groups
X_train_G1G2 = X_train[:, :20]
X_train_G3 = X_train[:, 20:]

X_test_G1G2 = X_test[:, :20]
X_test_G3 = X_test[:, 20:]

# Train individual logistic regression models
model_G1G2 = LogisticRegression(penalty='l2', solver='liblinear', max_iter=200)
model_G1G2.fit(X_train_G1G2, y_train)

model_G3 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=200)
model_G3.fit(X_train_G3, y_train)

# Combine predictions (simple averaging of probabilities)
proba_G1G2 = model_G1G2.predict_proba(X_test_G1G2)[:, 1]
proba_G3 = model_G3.predict_proba(X_test_G3)[:, 1]
final_proba = (proba_G1G2 + proba_G3) / 2
final_pred = (final_proba > 0.5).astype(int)

# Evaluate
accuracy = accuracy_score(y_test, final_pred)
f1 = f1_score(y_test, final_pred)
roc_auc = roc_auc_score(y_test, final_proba)

{
    "Accuracy": accuracy,
    "F1 Score": f1,
    "ROC AUC": roc_auc
}
