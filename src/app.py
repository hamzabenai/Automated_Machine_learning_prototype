import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import sys
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pipelines.main_pip import main_pipeline

def main():
    st.set_page_config(page_title="AutoML Pipeline", layout="wide")
    st.title("AutoML Pipeline")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    
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
            
            # Model type selection (user choice)
            model_type = st.radio(
                "Select model type",
                ["Classification", "Regression"],
                help="Choose whether to treat this as a classification or regression problem"
            )
            
            # Model selection based on user choice
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
            
            # Model selection
            model_name = st.selectbox(f"Select {model_type} Model", models)
            
            # Data type warning if mismatch detected
            if (model_type == "Classification" and 
                df[target_col].dtype not in ['object', 'category']):
                st.warning("Note: You've selected classification but the target appears numeric. "
                         "Ensure this is correct - numeric targets will be treated as class labels.")
            
            if (model_type == "Regression" and 
                df[target_col].dtype in ['object', 'category']):
                st.warning("Note: You've selected regression but the target appears categorical. "
                         "Ensure this is correct - categorical targets will be encoded as numeric.")
            
            if st.button("Run Pipeline"):
                with st.spinner("Training model..."):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Save temp file
                        temp_path = Path(temp_dir) / "temp_data.csv"
                        df.to_csv(temp_path, index=False)
                        
                        # Run pipeline with explicit model_type
                        result = main_pipeline(
                            data_path=str(temp_path),
                            target=target_col,
                            model_name=model_name,
                            model_path="model.pkl",
                            model_type=(model_type == "Classification")  # True for classification
                        )
                        
                        # Display results
                        if isinstance(result, dict):
                            st.success("Model trained successfully!")
                            
                            # Load evaluation results
                            try:
                                with open("model.pkl", "rb") as f:
                                    model = pickle.load(f)
                                
                                # Display performance metrics
                                st.subheader("Model Performance")
                                
                                if model_type == "Classification":
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Accuracy", f"{result['accuracy']:.2f}")
                                        st.metric("F1 Score", f"{result['f1_score']:.2f}")
                                    with col2:
                                        st.metric("Mean Cross-val Score", f"{result['mean_cross_val_score']:.2f}")
                                        st.metric("Cross-Val Score", f"{result['mean_cross_val_score']:.2f}")
                                else:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("R² Score", f"{result['r2_score']:.2f}")
                                        st.metric("MSE", f"{result['mean_squared_error']:.2f}")
                                    with col2:
                                        st.metric("RMSE", f"{result['mean_squared_error']**0.5:.2f}")
                                        st.metric("Cross-Val Score", f"{result['mean_cross_val_score']:.2f}")
                                
                                # Download button
                                with open("model.pkl", "rb") as f:
                                    st.download_button(
                                        label="Download Model",
                                        data=f,
                                        file_name="trained_model.pkl",
                                        mime="application/octet-stream"
                                    )
                                
                            except Exception as e:
                                st.error(f"Could not load model: {str(e)}")
                        else:
                            st.error(f"Model training failed: {result}")
                            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()