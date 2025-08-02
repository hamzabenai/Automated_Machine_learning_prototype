import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import sys
import tempfile
import numpy as np
import sklearn
import imblearn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pipelines.main_pip import main_pipeline
from strategies.PredictClass import PredictNow

def main():
    st.write(f"scikit-learn version: {sklearn.__version__}")
    st.write(f"imbalanced-learn version: {imblearn.__version__}")
    st.set_page_config(page_title="Automated Supervised Learning", layout="wide")
    st.title("Automated Supervised Learning")
    
    # Initialize session state
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None
    if 'proc_models' not in st.session_state:
        st.session_state.proc_models = None
    if 'prediction_data' not in st.session_state:
        st.session_state.prediction_data = None
    
    # File upload section
    uploaded_file = st.file_uploader("Upload your training dataset", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Show data preview
            with st.expander("Data Preview"):
                st.dataframe(df.head())
            
            # Column selection
            st.subheader("Feature Selection")
            selected_columns = st.multiselect(
                "Select columns to remove (optional)",
                df.columns,
                help="Remove any unnecessary columns before model training"
            )
            
            # Remove selected columns
            if selected_columns:
                df = df.drop(columns=selected_columns)
                st.success(f"Removed columns: {', '.join(selected_columns)}")
                st.dataframe(df.head())
            
            # Target selection
            target_col = st.selectbox("Select target variable", df.columns)
            
            # Model type selection
            model_type = st.radio(
                "Select Modeling Problem",
                ["Classification", "Regression"],
                help="Choose whether to treat this as a classification or regression problem"
            )
            
            # Model selection
            if model_type == "Classification":
                models = [
                    "RandomForestClassifier", 
                    "XGBClassifier", 
                    "LogisticRegression",
                    "KNeighborsClassifier", 
                    "DecisionTreeClassifier",
                    "SVC"
                ]
            else:
                models = [
                    "RandomForestRegressor", 
                    "XGBRegressor", 
                    "LinearRegression",
                    "KNeighborsRegressor", 
                    "DecisionTreeRegressor", 
                    "SVR"
                ]
            
            model_name = st.selectbox(f"Select {model_type} Model", models)
            
            # Data type warnings
            if (model_type == "Classification" and 
                df[target_col].dtype not in ['object', 'category']):
                st.warning("Note: You've selected classification but the target appears numeric.")
            
            if (model_type == "Regression" and 
                df[target_col].dtype in ['object', 'category']):
                st.warning("Note: You've selected regression but the target appears categorical.")
            
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir) / "temp_data.csv"
                        df.to_csv(temp_path, index=False)
                        
                        result = main_pipeline(
                            data_path=str(temp_path),
                            target=target_col,
                            model_name=model_name,
                            model_path="model.pkl",
                            model_type=(model_type == "Classification")
                        )
                        
                        if isinstance(result, dict):
                            st.success("Model trained successfully!")
                            st.session_state.trained_model = result['model']
                            st.session_state.selected_features = result['selected_feature']
                            st.session_state.proc_models = result['proc_models']
                            
                            # Display metrics
                            st.subheader("Model Performance")
                            if model_type == "Classification":
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Accuracy", f"{result['evaluation_results']['accuracy']:.2f}")
                                    st.metric("F1 Score", f"{result['evaluation_results']['f1_score']:.2f}")
                                with col2:
                                    st.metric("Mean Cross-val Score", f"{result['evaluation_results']['mean_cross_val_score']:.2f}")
                            else:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("R² Score", f"{result['evaluation_results']['r2_score']:.2f}")
                                    st.metric("MSE", f"{result['evaluation_results']['mean_squared_error']:.2f}")
                                with col2:
                                    st.metric("RMSE", f"{result['evaluation_results']['mean_squared_error']**0.5:.2f}")
                            
                            # Download model
                            with open("model.pkl", "rb") as f:
                                st.download_button(
                                    label="Download Model",
                                    data=f,
                                    file_name="trained_model.pkl",
                                    mime="application/octet-stream"
                                )
                        else:
                            st.error(f"Model training failed: {result}")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Prediction section (independent of training block)
    if st.session_state.trained_model is not None:
        st.subheader("Make Predictions")
        pred_file = st.file_uploader("Upload data for prediction", 
                                   type=["csv", "xlsx"], 
                                   key="pred_file_uploader")
        
        if pred_file is not None:
            try:
                if pred_file.name.endswith('.csv'):
                    pred_df = pd.read_csv(pred_file)
                else:
                    pred_df = pd.read_excel(pred_file)
                
                st.session_state.prediction_data = pred_df
                
                with st.expander("Prediction Data Preview"):
                    st.dataframe(pred_df.head())
                
                if st.button("Predict"):
                    with st.spinner("Making predictions..."):
                        predictor = PredictNow()
                        results = predictor.predict(
                            data=st.session_state.prediction_data,
                            model=st.session_state.trained_model,
                            features=st.session_state.selected_features,
                            proc_models=st.session_state.proc_models
                        )
                        
                        st.success("Predictions completed successfully!")
                        st.subheader("Prediction Results")
                        st.dataframe(results)
                        
                        csv = results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error processing prediction file: {str(e)}")

if __name__ == "__main__":
    main()