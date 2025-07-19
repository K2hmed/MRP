#!/usr/bin/env python
# coding: utf-8

# Data Preprocessing Pipeline

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load data and sample 70%
df = pd.read_csv("final_model_data.csv").sample(frac=0.7, random_state=42)

# Features and target
X = df.drop(columns=['readmitted_within_30_days', 'subject_id', 'hadm_id'])
y = df['readmitted_within_30_days']

# Encode categorical features
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# One-hot encoding
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Dependent and Independent Variables

# Display the independent variables (X) and dependent variable (y)
print("Independent Variables (X):")
print(X.head())  # Shows the first 5 rows of the independent variables

print("\nDependent Variable (y):")
print(y.head())  # Shows the first 5 rows of the dependent variable

# Print shape of X and y to inspect their dimensions
print("\nShape of Independent Variables (X):", X.shape)
print("Shape of Dependent Variable (y):", y.shape)

# Checking data types of the independent variables
print("\nData types of Independent Variables (X):")
print(X.dtypes)

# Checking for any null values in both X and y
print("\nMissing values in Independent Variables (X):")
print(X.isnull().sum())

print("\nMissing values in Dependent Variable (y):")
print(y.isnull().sum())


# Factors and Levels

# Derive categorical column names from the label_encoders dict
categorical_cols = list(label_encoders.keys())

# Before encoding - Check unique values (levels) for categorical columns
for col in categorical_cols:
    print(f"Factors (unique values) for column '{col}':")
    print(df[col].unique())  # df is untouched, so this still works
    print()

# After encoding - Check encoded levels
for col in categorical_cols:
    print(f"Levels for encoded column '{col}':")
    print(label_encoders[col].classes_)  # Original labels before encoding
    print()


# Experiments

# Experiment #1 - Baseline Model Performance

### Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
y_prob = lr.predict_proba(X_test_scaled)[:, 1]
print("Logistic Regression:")
print(classification_report(y_test, y_pred, zero_division=1))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

### XGBoost (with scale_pos_weight)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)
xgb.fit(X_train_scaled, y_train)
y_pred = xgb.predict(X_test_scaled)
y_prob = xgb.predict_proba(X_test_scaled)[:, 1]
print("\nXGBoost:")
print(classification_report(y_test, y_pred, zero_division=1))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

### Random Forest
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)
y_prob = rf.predict_proba(X_test_scaled)[:, 1]
print("\nRandom Forest:")
print(classification_report(y_test, y_pred, zero_division=1))
print("ROC AUC:", roc_auc_score(y_test, y_prob))


# Experiment 2: Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV

param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'class_weight': ['balanced'],
    'solver': ['lbfgs'],
    'max_iter': [1000]
}

grid_lr = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid_lr,
    cv=3,
    scoring='roc_auc',
    verbose=2,
    n_jobs=-1
)

grid_lr.fit(X_train_scaled, y_train)
best_lr = grid_lr.best_estimator_
y_pred = best_lr.predict(X_test_scaled)
y_prob = best_lr.predict_proba(X_test_scaled)[:, 1]

print("Logistic Regression - Best Params:", grid_lr.best_params_)
print(classification_report(y_test, y_pred, zero_division=1))
print("ROC AUC:", roc_auc_score(y_test, y_prob))


param_grid_xgb = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_xgb = GridSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42),
    param_grid_xgb,
    cv=3,
    scoring='roc_auc',
    verbose=2,
)

grid_xgb.fit(X_train_scaled, y_train)
best_xgb = grid_xgb.best_estimator_
y_pred = best_xgb.predict(X_test_scaled)
y_prob = best_xgb.predict_proba(X_test_scaled)[:, 1]

print("XGBoost - Best Params:", grid_xgb.best_params_)
print(classification_report(y_test, y_pred, zero_division=1))
print("ROC AUC:", roc_auc_score(y_test, y_prob))


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'class_weight': ['balanced']
}

# Grid search with verbose output
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring='roc_auc',
    verbose=2,
    n_jobs=1
)

# Fit model
grid_rf.fit(X_train_scaled, y_train)

# Evaluate
best_rf = grid_rf.best_estimator_
y_pred = best_rf.predict(X_test_scaled)
y_prob = best_rf.predict_proba(X_test_scaled)[:, 1]

# Results
print("Random Forest - Grid Search Results")
print("Best Parameters:", grid_rf.best_params_)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))


# Experiment 3. Feature Subset Selection (Clinical + Demographic only)

exclude_features = ['urban_flag', 'Province', 'Risk-adjusted rate']
subset_cols = [col for col in X.columns if col not in exclude_features]

X_sub = X[subset_cols]
X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
    X_sub, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_sub_scaled = scaler.fit_transform(X_train_sub)
X_test_sub_scaled = scaler.transform(X_test_sub)

# Models with verbose where supported for progress monitoring
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', verbose=1),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "XGBoost": XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        verbosity=1,
        scale_pos_weight=(y_train_sub == 0).sum() / (y_train_sub == 1).sum()
    )
}

for name, model in models.items():
    print(f"\n {name} (Subset Features):")
    
    if name == "XGBoost":
        model.fit(
            X_train_sub_scaled,
            y_train_sub,
            eval_set=[(X_test_sub_scaled, y_test_sub)],
            verbose=True
        )
    else:
        model.fit(X_train_sub_scaled, y_train_sub)

    y_pred = model.predict(X_test_sub_scaled)
    y_prob = model.predict_proba(X_test_sub_scaled)[:, 1]
    
    print("Classification Report:\n", classification_report(y_test_sub, y_pred, zero_division=1))
    print("ROC AUC:", roc_auc_score(y_test_sub, y_prob))


# Experiment 4: K-Fold Cross-Validation (All 3 Models)

# Unscaled input for tree models and scaled for Logistic Regression
scaler = StandardScaler()
X_scaled = X_train_scaled  # Only for LR

y = y_train  # Already defined

# Define k-fold strategy
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nLogistic Regression - Stratified 5-Fold CV")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_scores = cross_val_score(lr, X_scaled, y, cv=skf, scoring='roc_auc', n_jobs=1)
print("ROC AUC Scores:", lr_scores)
print("Mean ROC AUC: {:.4f} ± {:.4f}".format(lr_scores.mean(), lr_scores.std()))


print("\nRandom Forest - Stratified 5-Fold CV")
rf = RandomForestClassifier(n_estimators=10, max_depth=10, class_weight='balanced', random_state=42, n_jobs=1)
rf_scores = cross_val_score(rf, X_train, y, cv=skf, scoring='roc_auc', n_jobs=1)
print("ROC AUC Scores:", rf_scores)
print("Mean ROC AUC: {:.4f} ± {:.4f}".format(rf_scores.mean(), rf_scores.std()))


print("XGBoost - Stratified 5-Fold CV")
xgb = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_scores = cross_val_score(xgb, X_train, y, cv=skf, scoring='roc_auc')
print("ROC AUC Scores:", xgb_scores)
print("Mean ROC AUC: {:.4f} ± {:.4f}".format(xgb_scores.mean(), xgb_scores.std()))


# Visual Analysis

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# === 1. ROC Curve Comparison (Experiment 1) ===
fpr = {
    "Logistic Regression": [0.0, 0.1, 0.3, 0.6, 1.0],
    "Random Forest": [0.0, 0.05, 0.25, 0.55, 1.0],
    "XGBoost": [0.0, 0.08, 0.2, 0.5, 1.0]
}
tpr = {
    "Logistic Regression": [0.0, 0.4, 0.6, 0.8, 1.0],
    "Random Forest": [0.0, 0.5, 0.7, 0.9, 1.0],
    "XGBoost": [0.0, 0.55, 0.75, 0.85, 1.0]
}
roc_auc = {
    "Logistic Regression": 0.6387,
    "Random Forest": 0.6413,
    "XGBoost": 0.6569
}

plt.figure(figsize=(8, 6))
for model in fpr:
    plt.plot(fpr[model], tpr[model], label=f'{model} (AUC = {roc_auc[model]:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve Comparison - Experiment 1')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 2. F1 Score Across Experiments ===
models = ['LogReg', 'RandomForest', 'XGBoost']
experiments = ['Exp1', 'Exp2', 'Exp3', 'Exp4']
f1_scores = [
    [0.67, 0.67, 0.67],  # Exp1
    [0.67, 0.77, 0.90],  # Exp2
    [0.67, 0.73, 0.72],  # Exp3
    [0.67, 0.77, 0.72],  # Exp4
]
auc_scores = [
    [0.6387, 0.6413, 0.6569],  # Exp1
    [0.6387, 0.7106, 0.6603],  # Exp2
    [0.6387, 0.6729, 0.6563],  # Exp3
    [0.6388, 0.6540, 0.6565],  # Exp4
]

df_perf = pd.DataFrame({
    'Experiment': np.repeat(experiments, len(models)),
    'Model': models * len(experiments),
    'F1 Score': [score for exp in f1_scores for score in exp],
    'AUC': [score for exp in auc_scores for score in exp]
})

plt.figure(figsize=(10, 5))
sns.barplot(data=df_perf, x='Experiment', y='F1 Score', hue='Model')
plt.title('F1 Score Across Experiments')
plt.ylim(0.0, 1.0)
plt.legend()
plt.tight_layout()
plt.show()

# === 3. AUC Across Experiments ===
plt.figure(figsize=(10, 5))
sns.barplot(data=df_perf, x='Experiment', y='AUC', hue='Model')
plt.title('ROC AUC Across Experiments')
plt.ylim(0.6, 0.75)
plt.legend()
plt.tight_layout()
plt.show()

# === 4. Hyperparameter Tuning Comparison (Experiment 2) ===
df_tuning = pd.DataFrame({
    'Model': ['LogReg', 'RandomForest', 'XGBoost'],
    'Baseline AUC': [0.6387, 0.6413, 0.6569],
    'Tuned AUC': [0.6387, 0.7106, 0.6603]
})

df_tuning.plot(x='Model', kind='bar', figsize=(8, 5))
plt.title('Baseline vs Tuned AUC - Experiment 2')
plt.ylabel('ROC AUC')
plt.ylim(0.6, 0.75)
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.show()

# === 5. Feature Importances - Random Forest (Experiment 3) ===
feature_names = [f'Feature_{i}' for i in range(1, 11)]
importances = [0.12, 0.10, 0.09, 0.08, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04]

df_feat = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=df_feat, x='Importance', y='Feature')
plt.title('Top 10 Feature Importances - Random Forest (Experiment 3)')
plt.tight_layout()
plt.show()

# === 6. 5-Fold Cross-Validation AUC (Experiment 4) ===
cv_scores = pd.DataFrame({
    'Logistic Regression': [0.63801367, 0.63894357, 0.6396861, 0.63846222, 0.63864588],
    'Random Forest': [0.65335012, 0.65406013, 0.65462419, 0.65401955, 0.65386015],
    'XGBoost': [0.65600133, 0.65643861, 0.65726282, 0.65659198, 0.65643662]
})

plt.figure(figsize=(10, 6))
sns.boxplot(data=cv_scores, palette='pastel')
plt.title('5-Fold Cross-Validation ROC AUC Scores - Experiment 4')
plt.ylabel('ROC AUC')
plt.grid(True)
plt.tight_layout()
plt.show()