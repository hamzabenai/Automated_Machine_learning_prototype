# Automated Supervised Learning Pipeline

## Overview

This project provides an automated pipeline for supervised learning tasks (both classification and regression) with a Streamlit-based web interface. The system handles data preprocessing, model training, evaluation, validation, and prediction in a structured and automated manner.

## Features

- **User-friendly Web Interface**: Built with Streamlit for easy interaction
- **Automated Data Processing**: Handles missing values, outliers, encoding, scaling, and feature selection
- **Model Training**: Supports multiple classification and regression algorithms
- **Model Evaluation**: Comprehensive performance metrics for both training and testing
- **Prediction System**: Make predictions on new data with the trained model
- **Validation System**: Ensures model reliability before deployment

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
    ```bash
    streamlit run app.py
    ```

## Usage

1. Upload Data: Upload your training dataset (CSV or Excel format)
2. Data Preprocessing:
  - Remove unnecessary columns
  - Select target variable
  - Choose problem type (Classification/Regression)

3. Model Selection: Choose from available algorithms
4. Training: Click "Train Model" to start the automated pipeline
5. Evaluation: View performance metrics after training
6. Prediction: Upload new data to make predictions with the trained model

## Pipeline Architecture
1. Data Ingestion `IngestClass.py`
  - Loads data from CSV or Excel files
  - Validates the presence of target column
  - Returns pandas DataFrame
2. Data Cleaning `CleanClass.py`, `cleanData.py`
- **Identifier Removal**: Drops columns with single unique value or all unique values

- **Missing Value Handling:** For small datasets (<2000 rows): Imputes with mean/mode, For larger datasets: Drops columns with >20% missing values
- **Outlier Treatment:** Small datasets: IQR method, Medium datasets: Local Outlier Factor, Large datasets: Isolation Forest
- **Imbalanced Data Handling: (Classification)** Upsamples minority class when class imbalance > 75%
- **Encoding:** Label encoding for categorical features
- **Scaling:** Robust scaling for numerical features
- **Feature Selection:** Mutual information-based feature selection
- **Data Splitting:** 80-20 train-test split with stratification

3. Model Training `TrainClass.py`, `trainModel.py`
- **Classification Models:** Random Forest, XGBoost, K-Nearest Neighbors, SVM, Logistic Regression, Decision Tree
- **Regression Models:** Random Forest, XGBoost, K-Nearest Neighbors, SVM, Linear Regression, Decision Tree

4. Model Evaluation `EvaluateClass.py`, `evaluateModel.py`
- **Classification Metrics:** Accuracy, F1 Score, Confusion Matrix, Cross-validation scores
- **Regression Metrics:** Mean Squared Error, R² Score, Cross-validation scores

5. Model Validation `ValidateClass.py `, `validateModel.py`
- Consistency between training and testing performance
- Absolute performance thresholds
- Cross-validation stability

6. Model Export `ExportClass.py`, `exportModel.py`
- Saves trained model as pickle file
- Includes preprocessing objects (encoders, scalers)

7. Prediction System `PredictClass.py`
- Applies same preprocessing to new data
- Uses trained model to make predictions
- Returns predictions with original data

## Design Patterns
The system uses the Strategy Pattern extensively, with abstract base classes defining interfaces for : Data cleaning strategies, Model training strategies, Evaluation strategies, Validation strategies, Export strategies.
This design allows for easy extension and modification of individual components without affecting the overall pipeline.

## Future Enhancements
- Add hyperparameter tuning capabilities
- Include more feature engineering options
- Add support for time series data
- Implement model interpretation tools (SHAP, LIME)
- Add deployment options (Docker, cloud services)

## Contributors
Hamza Benai
