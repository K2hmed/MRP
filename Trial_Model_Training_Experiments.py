#!/usr/bin/env python
# coding: utf-8

# Data Preprocessing Pipeline

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Load and sample the data
df = pd.read_csv("final_model_data.csv").sample(frac=0.5, random_state=42)

# Define target and drop identifiers
X = df.drop(columns=['readmitted_within_30_days', 'subject_id', 'hadm_id'])
y = df['readmitted_within_30_days']

# Label encode categorical variables
categorical_cols = X.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# One-hot encode after label encoding
X_encoded = pd.get_dummies(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Dependent and Independent Variables

# Display the independent variables (X) and dependent variable (y)
print("Independent Variables (X):")
print(X.head())  # Shows the first 5 rows of the independent variables

print("\nDependent Variable (y):")
print(y.head())  # Shows the first 5 rows of the dependent variable

# You can also print the shape of X and y to see their dimensions
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

# Before encoding - Check unique values (levels) for categorical columns
for col in categorical_cols:
    print(f"Factors (unique values) for column '{col}':")
    print(df[col].unique())
    print()

# After encoding - Check encoded values for categorical columns
for col in categorical_cols:
    print(f"Levels for encoded column '{col}':")
    # Reverse the encoding to get original categories
    original_categories = label_encoders[col].classes_
    print(original_categories)
    print()


# Experiments

# Experiment 1: Baseline Model Training

# Logistic Regression Model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Create and train the model
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Print classification report with zero_division set to 1
print("📊 Logistic Regression Performance:")
print(classification_report(y_test, y_pred, zero_division=1))

# Print ROC AUC score
print("ROC AUC:", roc_auc_score(y_test, y_prob))


# XGBoost Model

from xgboost import XGBClassifier

model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("📊 XGBoost Performance:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))


# Random Forest Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("📊 Random Forest Performance:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))


# Experiment 2: Hyperparameter Tuning
# (Logistic Regression)

param_grid_lr = {'C': [0.1, 1.0, 10]}
grid_lr = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42), param_grid_lr, cv=3, scoring='roc_auc', verbose=2, n_jobs=-1
)
grid_lr.fit(X_train_scaled, y_train)

# Best model and predictions
best_lr = grid_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test_scaled)
y_prob_lr = best_lr.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("Best Parameters:", grid_lr.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_test, y_prob_lr))


# (XGBoost)

param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.2]
}

grid_xgb = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=42),
                        param_grid, cv=3, scoring='roc_auc')
grid_xgb.fit(X_train_scaled, y_train)

best_xgb = grid_xgb.best_estimator_
y_pred = best_xgb.predict(X_test_scaled)
y_prob = best_xgb.predict_proba(X_test_scaled)[:, 1]

print("XGBoost - Grid Search Results")
print("Best Parameters:", grid_xgb.best_params_)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))


# (Random Forest)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
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
subset_cols = [col for col in X_encoded.columns if col not in exclude_features]

X_sub = X_encoded[subset_cols]
X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
    X_sub, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_sub_scaled = scaler.fit_transform(X_train_sub)
X_test_sub_scaled = scaler.transform(X_test_sub)

# Models with verbose where supported
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, verbose=1),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        verbosity=1
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
    
    print("Classification Report:\n", classification_report(y_test_sub, y_pred))
    print("ROC AUC:", roc_auc_score(y_test_sub, y_prob))


# Experiment 4: K-Fold Cross-Validation (All 3 Models)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Manually perform cross-validation with verbose output
    scores = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_encoded, y), 1):
        X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit the model on the current fold's training data
        model.fit(X_train, y_train)
        
        # Predict on the test set
        score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        scores.append(score)
        
        print(f"  Fold {fold} ROC AUC: {score:.4f}")
    
    # After cross-validation, calculate mean and std
    print(f"\n{name} - 5-Fold ROC AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

