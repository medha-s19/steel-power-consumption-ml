from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_preprocessing import TARGET_COLUMN, load_data, preprocess_data
from src.model_training import (
    load_model_bundle,
    prepare_training_data,
    save_model,
    train_model,
)
from src.predict import predict_from_raw


DATA_PATH = Path("data/Steel_industry_data.csv")
MODEL_PATH = Path("models/model.pkl")
MODEL_OPTIONS = ["Linear Regression", "Random Forest"]


st.set_page_config(page_title="Steel Energy Predictor", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: #f6f8fb;
        }
        .block-container {
            max-width: 1100px;
            padding-top: 1.5rem;
        }
        [data-testid="stSidebar"] {
            background: #0f172a;
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0;
        }
        .app-header {
            background: linear-gradient(135deg, #0f172a, #2563eb);
            border-radius: 16px;
            padding: 1.4rem 1.8rem;
            margin-bottom: 1.4rem;
            color: white;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.18);
        }
        .app-header h1 {
            margin: 0;
            font-size: 2.2rem;
            font-weight: 800;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dbe4ee;
            border-radius: 12px;
            padding: 0.75rem 1rem;
        }
        div[data-testid="stMetric"] label {
            color: #4b5563 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #111827 !important;
        }
        div[data-testid="stMetric"] > div {
            color: #111827 !important;
        }
        .info-card {
            background: #ffffff;
            border: 1px solid #dbe4ee;
            border-radius: 12px;
            padding: 0.95rem 1rem;
            min-height: 96px;
        }
        .info-label {
            color: #4b5563;
            font-size: 0.9rem;
            margin-bottom: 0.55rem;
        }
        .info-value {
            color: #111827;
            font-size: 1.45rem;
            font-weight: 650;
        }
        .stButton > button {
            border-radius: 10px;
            font-weight: 650;
        }
        /* Fix input labels visibility */
        label[data-testid="stWidgetLabel"] p {
            color: #111827 !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        return load_data(DATA_PATH)
    return pd.DataFrame()


def show_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <h1>Steel Energy Consumption Predictor</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_metrics(metrics: dict) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{metrics['mse']:.4f}")
    col2.metric("RMSE", f"{metrics['rmse']:.4f}")
    col3.metric("R2 Score", f"{metrics['r2']:.4f}")


def get_saved_model_info() -> tuple[str | None, dict]:
    if not MODEL_PATH.exists():
        return None, {}

    try:
        artifact = load_model_bundle(MODEL_PATH)
    except Exception:
        return None, {}

    return artifact.get("model_name"), artifact.get("metrics", {})


def prediction_form(frame: pd.DataFrame) -> dict:
    input_data = {}
    feature_frame = frame.drop(columns=[TARGET_COLUMN], errors="ignore")

    with st.form("prediction_form"):
        for column in feature_frame.columns:
            if pd.api.types.is_numeric_dtype(feature_frame[column]):
                input_data[column] = st.number_input(
                    column,
                    value=float(feature_frame[column].median()),
                )
            else:
                options = sorted(feature_frame[column].dropna().astype(str).unique()) or [""]
                input_data[column] = st.selectbox(column, options)

        submitted = st.form_submit_button("Predict")

    return input_data if submitted else {}


show_header()

frame = load_dataset()
if frame.empty:
    st.error("Dataset not found. Please check data/Steel_industry_data.csv.")
    st.stop()

try:
    cleaned_frame = preprocess_data(frame)
    features, target = prepare_training_data(frame, target_column=TARGET_COLUMN)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

saved_model_name, saved_metrics = get_saved_model_info()

page = st.sidebar.radio(
    "Navigation",
    ["Dataset", "Train Model", "Prediction"],
)

if page == "Dataset":
    st.subheader("Dataset Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(frame):,}")
    col2.metric("Columns", f"{len(frame.columns):,}")
    col3.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">Target</div>
            <div class="info-value">{TARGET_COLUMN}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(frame.head(15), width="stretch")

elif page == "Train Model":
    st.subheader("Model Training")
    model_choice = st.selectbox("Select Model", MODEL_OPTIONS)

    if saved_model_name:
        st.info(f"Current saved model: {saved_model_name}")
    else:
        st.info("No trained model saved yet.")

    if st.button("Train Model", type="primary"):
        try:
            model, metrics = train_model(features, target, model_name=model_choice)
            save_model(
                model,
                MODEL_PATH,
                feature_columns=list(features.columns),
                metrics=metrics,
                model_name=model_choice,
            )
            st.success(f"{model_choice} trained successfully.")
            show_metrics(metrics)
        except ValueError as exc:
            st.error(str(exc))
    elif saved_metrics:
        st.write("Last training metrics")
        show_metrics(saved_metrics)

elif page == "Prediction":
    st.subheader("Prediction")

    if not MODEL_PATH.exists():
        st.warning("Train a model first before making predictions.")
        st.stop()

    try:
        artifact = load_model_bundle(MODEL_PATH)
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        st.stop()

    if not artifact.get("feature_columns"):
        st.warning("Model metadata is missing. Please train the model again.")
        st.stop()

    st.info(f"Using saved model: {artifact.get('model_name', 'Unknown')}")
    input_data = prediction_form(cleaned_frame)

    if input_data:
        try:
            prediction = predict_from_raw(MODEL_PATH, input_data)
            st.success(f"Predicted Energy Usage: {prediction:.2f} kWh")
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))