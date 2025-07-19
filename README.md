# MRP
# 📄 30-Days Hospital Readmission Prediction Using EHR Data

## 1. Abstract

The admission of patients to hospitals places considerable financial strain on healthcare systems while also serving as a warning signal regarding the quality of care delivered during the initial hospital visit.

This research develops and validates a model to predict **30-day hospital readmissions** by merging Electronic Health Record (EHR) clinical information with **socioeconomic factors**. 

The study utilizes a variety of publicly available datasets, including:

- **MIMIC IV Database**
- **eICU Collaborative Research Database**
- **CIHI Open Source Dataset**
- **ODHF OpenSource Dataset**

The objective is to determine whether incorporating socioeconomic information enhances predictive performance and helps identify patients at elevated risk. The study adopts modern **machine learning practices** to build interpretable models that support healthcare professionals in both clinical decision-making and resource allocation.

## 2. Problem Definition

Hospital readmissions within 30 days post-discharge pose significant challenges:

- ⚕️ **Clinical Concerns**: May indicate suboptimal inpatient care or inadequate discharge planning.  
- 💸 **Financial Impact**: Lead to increased healthcare costs and potential penalties (e.g., under CMS policies).  
- ⚖️ **Equity Issues**: Emerging evidence suggests socioeconomic factors contribute to readmission risks—factors that current models often fail to capture.

# Hospital Readmission Prediction Pipeline (TMU MRP)

This repository contains three main Python scripts used in the Major Research Project (MRP) at Toronto Metropolitan University (TMU). The project focuses on predictive modeling for 30-day hospital readmissions using clinical, demographic, and geographic data from open-source datasets (MIMIC-IV, eICU, CIHI, and ODHF).

## File Descriptions

### 1. `EDA.py`
**Purpose**:  
Performs data loading, cleaning, preprocessing, and feature engineering using the MIMIC-IV, eICU, CIHI, and ODHF datasets.  
Generates exploratory visualizations (e.g., univariate, bivariate, multivariate, geospatial) and produces the final merged dataset `final_model_data.csv` for modeling.

### 2. `Trial_Model_Training_Experiments.py`
**Purpose**:  
Performs trial model training and evaluation on a 50% sample of the dataset.  
Includes:
- Baseline modeling with Logistic Regression, Random Forest, and XGBoost
- Hyperparameter tuning via GridSearchCV
- Feature subset experiment (clinical + demographic features only)
- Stratified 5-Fold Cross-Validation

### 3. `Final_Model_Training_Experiments.py`
**Purpose**:  
Runs the full experimental pipeline on a 70% sample of the dataset.  
Includes:
- Baseline model training (Logistic Regression, Random Forest, XGBoost)
- Hyperparameter tuning
- Feature subset experiment (clinical + demographic features only)
- Stratified 5-Fold Cross-Validation
- Final visualization of model performance (ROC curves, AUC/F1 bar charts, feature importance, CV results)

**Note**:  
Some visualizations use hardcoded metric values based on actual model output to reduce runtime on large datasets (>5.6GB).


