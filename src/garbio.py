import gradio as gr
import pandas as pd
import tempfile
import pickle
import numpy as np
from pathlib import Path
import sys

# Add root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from pipelines.main_pip import main_pipeline
from strategies.PredictClass import PredictNow

# Globals to hold state
state = {
    "trained_model": None,
    "selected_features": None,
    "proc_models": None
}

def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        else:
            df = pd.read_excel(file.name)
        return df, ""
    except Exception as e:
        return None, str(e)

def train_model(file, target_col, model_type, model_name, remove_cols):
    try:
        df, err = load_data(file)
        if err:
            return None, None, None, f"Error loading file: {err}"

        if remove_cols:
            df = df.drop(columns=remove_cols)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "temp.csv"
            df.to_csv(tmp_path, index=False)

            result = main_pipeline(
                data_path=str(tmp_path),
                target=target_col,
                model_name=model_name,
                model_path="model.pkl",
                model_type=(model_type == "Classification")
            )

        if isinstance(result, dict):
            state["trained_model"] = result['model']
            state["selected_features"] = result['selected_feature']
            state["proc_models"] = result['proc_models']

            metrics = result['evaluation_results']
            if model_type == "Classification":
                summary = {
                    "Accuracy": round(metrics['accuracy'], 2),
                    "F1 Score": round(metrics['f1_score'], 2),
                    "CV Score": round(metrics['mean_cross_val_score'], 2)
                }
            else:
                summary = {
                    "R2 Score": round(metrics['r2_score'], 2),
                    "MSE": round(metrics['mean_squared_error'], 2),
                    "RMSE": round(metrics['mean_squared_error']**0.5, 2)
                }

            with open("model.pkl", "rb") as f:
                model_bytes = f.read()

            return summary, result['selected_feature'], model_bytes, "Model trained successfully!"
        else:
            return None, None, None, f"Model training failed: {result}"

    except Exception as e:
        return None, None, None, str(e)

def predict_data(file):
    try:
        if state["trained_model"] is None:
            return None, "Model not trained yet."

        df, err = load_data(file)
        if err:
            return None, f"Error loading file: {err}"

        predictor = PredictNow()
        results = predictor.predict(
            data=df,
            model=state["trained_model"],
            features=state["selected_features"],
            proc_models=state["proc_models"]
        )
        return results, "Predictions complete."
    except Exception as e:
        return None, str(e)

# UI
def gradio_app():
    with gr.Blocks() as app:
        gr.Markdown("# 🧠 Automated Supervised Learning")

        with gr.Tab("Train Model"):
            train_file = gr.File(label="Upload training data (.csv or .xlsx)")
            remove_cols = gr.Textbox(label="Columns to remove (comma-separated)", placeholder="id, timestamp")
            target_col = gr.Textbox(label="Target Column")
            model_type = gr.Radio(["Classification", "Regression"], label="Model Type", value="Classification")
            model_name = gr.Dropdown(
                choices=[
                    "RandomForestClassifier", "XGBClassifier", "LogisticRegression",
                    "KNeighborsClassifier", "DecisionTreeClassifier", "SVC",
                    "RandomForestRegressor", "XGBRegressor", "LinearRegression",
                    "KNeighborsRegressor", "DecisionTreeRegressor", "SVR"
                ],
                label="Select Model"
            )
            train_btn = gr.Button("Train Model")
            metrics_output = gr.JSON(label="Training Metrics")
            selected_feats = gr.Textbox(label="Selected Features")
            download_model = gr.File(label="Download Trained Model")
            train_status = gr.Textbox(label="Status")

            def on_train(file, target_col, mtype, mname, rem_cols):
                cols = [c.strip() for c in rem_cols.split(",")] if rem_cols else []
                return train_model(file, target_col, mtype, mname, cols)

            train_btn.click(
                on_train,
                inputs=[train_file, target_col, model_type, model_name, remove_cols],
                outputs=[metrics_output, selected_feats, download_model, train_status]
            )

        with gr.Tab("Predict"):
            pred_file = gr.File(label="Upload data to predict")
            pred_btn = gr.Button("Predict")
            pred_output = gr.Dataframe()
            pred_status = gr.Textbox(label="Status")

            pred_btn.click(
                predict_data,
                inputs=[pred_file],
                outputs=[pred_output, pred_status]
            )

    return app

app = gradio_app()

if __name__ == "__main__":
    app.launch()
