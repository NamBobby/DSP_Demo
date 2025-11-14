import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================
# CONFIG
# ============================

# Paths (relative to project root)
BASE_MODEL_PATH = "models/logistic_regression_base.pkl"
ICA_MODEL_PATH  = "models/logistic_regression_ica.pkl"
LABEL_MAP_PATH  = "results/label_map.json"

# Number of DSP features per epoch
N_FEATURES = 24

# Optional demo feature file (one line of 24 features)
DEMO_FEATURE_PATH = "data_demo/demo_sample.csv"


# ============================
# LOADERS
# ============================

@st.cache_resource
def load_models_and_labels():
    """Load BASE & ICA models and label mapping from disk."""
    model_base = joblib.load(BASE_MODEL_PATH)
    model_ica = joblib.load(ICA_MODEL_PATH)

    with open(LABEL_MAP_PATH, "r") as f:
        label_map_raw = json.load(f)

    # Expecting {"id2label": {...}, "label2id": {...}} or a flat dict
    if "id2label" in label_map_raw:
        id2label_raw = label_map_raw["id2label"]
    else:
        id2label_raw = label_map_raw

    # Normalize keys to int
    id2label = {int(k): v for k, v in id2label_raw.items()}

    return model_base, model_ica, id2label


@st.cache_resource
def load_demo_sample():
    """Load a demo feature row from data_demo/demo_sample.csv (optional)."""
    try:
        df_demo = pd.read_csv(DEMO_FEATURE_PATH, header=None)
        if df_demo.shape[1] != N_FEATURES:
            return None
        # use first row
        x_demo = df_demo.iloc[0].values.reshape(1, -1)
        return x_demo
    except Exception:
        return None


model_base, model_ica, id2label = load_models_and_labels()
demo_sample = load_demo_sample()


# ============================
# HELPER FUNCTIONS
# ============================

def parse_feature_input(text: str):
    """
    Parse a comma-separated string into a numpy array of shape (1, N_FEATURES).
    Example input: "0.1, -0.23, 1.5, ..."
    """
    try:
        parts = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
        if len(parts) != N_FEATURES:
            st.error(f"You must enter exactly {N_FEATURES} values, but got {len(parts)}.")
            return None
        return np.array(parts, dtype=float).reshape(1, -1)
    except ValueError:
        st.error("Could not parse all values as floats. Please check your input.")
        return None


def get_prediction(model, x):
    """Return predicted class id, label, and probability vector (if available)."""
    y_pred_id = model.predict(x)[0]

    # Try to get probabilities (if model supports predict_proba)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
    else:
        # Fallback: pseudo one-hot probabilities for visualization
        n_classes = len(id2label)
        probs = np.zeros(n_classes)
        probs[int(y_pred_id)] = 1.0

    y_label = id2label[int(y_pred_id)]
    return y_pred_id, y_label, probs


def build_prob_dataframe(probs):
    """Build a DataFrame with Stage & Probability for bar chart."""
    class_ids = list(id2label.keys())
    class_labels = [id2label[c] for c in class_ids]
    prob_values = [probs[c] if c < len(probs) else 0.0 for c in class_ids]

    df = pd.DataFrame({
        "Stage": class_labels,
        "Probability": prob_values
    })
    return df.set_index("Stage")


# ============================
# STREAMLIT UI
# ============================

st.set_page_config(page_title="EEG Sleep Stage Demo", page_icon="🧠")

st.title("🧠 EEG Sleep Stage Classification Demo")
st.markdown(
    """
This app demonstrates our DSP + ML pipeline for **EEG sleep stage classification**.

Pipeline overview:
1. EEG signals are preprocessed using Digital Signal Processing (DSP): filtering, artifact removal (ICA), and feature extraction.
2. For each 30-second epoch, we compute a **24-dimensional feature vector**.
3. A trained ML model (Logistic Regression) predicts the **sleep stage label** (Wake, N1, N2, N3, REM).

In this demo, we load the trained models and:
- allow you to test **random / manual inputs**, and  
- upload an **external feature file** that is *not part of the training dataset*.
"""
)

# Choose which model to use
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.radio(
    "Choose feature set / model:",
    options=["BASE model (no ICA)", "ICA model (with artifact removal)"],
)

if model_choice.startswith("BASE"):
    current_model = model_base
    st.sidebar.info("Using model trained on BASE features (without ICA).")
else:
    current_model = model_ica
    st.sidebar.info("Using model trained on ICA-cleaned features.")

st.markdown("---")


# ============================
# SECTION 1 — QUICK DEMO INPUTS
# ============================

st.header("⚡ Quick Demo Inputs")

input_mode = st.selectbox(
    "How do you want to provide the feature vector?",
    options=[
        "Random demo vector (for UI testing)",
        "Manual input (24 comma-separated values)",
        "Demo sample from data_demo/demo_sample.csv",
    ],
)

if input_mode.startswith("Random demo"):
    st.caption(
        "A random feature vector is generated just to showcase the prediction pipeline. "
        "In a real scenario, this vector should come from the DSP feature extraction."
    )
    if st.button("Run prediction on random vector"):
        x_demo = np.random.randn(1, N_FEATURES)  # purely synthetic
        y_id, y_label, probs = get_prediction(current_model, x_demo)

        st.subheader("Prediction")
        st.write(f"**Predicted class ID:** `{y_id}`")
        st.write(f"**Predicted sleep stage:** `{y_label}`")

        st.subheader("Class probabilities")
        df_probs = build_prob_dataframe(probs)
        st.bar_chart(df_probs)

elif input_mode.startswith("Manual input"):
    st.caption(
        f"Please paste **{N_FEATURES} numeric values** separated by commas. "
        "These should correspond to the 24 DSP features for a single EEG epoch."
    )
    user_text = st.text_area(
        "Feature vector (comma-separated):",
        placeholder="e.g. 0.12, -0.34, 1.25, ... (24 values total)",
        height=120,
    )

    if st.button("Run prediction on manual input"):
        x_manual = parse_feature_input(user_text)
        if x_manual is not None:
            y_id, y_label, probs = get_prediction(current_model, x_manual)

            st.subheader("Prediction")
            st.write(f"**Predicted class ID:** `{y_id}`")
            st.write(f"**Predicted sleep stage:** `{y_label}`")

            st.subheader("Class probabilities")
            df_probs = build_prob_dataframe(probs)
            st.bar_chart(df_probs)

else:
    st.caption(
        "Use a pre-defined demo feature vector stored in `data_demo/demo_sample.csv`. "
        "This simulates an unseen external input prepared before the presentation."
    )
    if demo_sample is None:
        st.error(
            "Could not load demo_sample.csv or the number of features is not 24. "
            "Please check data_demo/demo_sample.csv."
        )
    else:
        if st.button("Run prediction on demo_sample.csv"):
            y_id, y_label, probs = get_prediction(current_model, demo_sample)

            st.subheader("Prediction (Demo Sample)")
            st.write("**Source file:** `data_demo/demo_sample.csv`")
            st.write(f"**Predicted class ID:** `{y_id}`")
            st.write(f"**Predicted sleep stage:** `{y_label}`")

            st.subheader("Class probabilities")
            df_probs = build_prob_dataframe(probs)
            st.bar_chart(df_probs)

            st.subheader("Conclusion")
            st.success(
                f"The demo feature vector is classified as **{y_label}**. "
                "This shows that the trained model can process a pre-computed DSP feature file "
                "and automatically assign a sleep stage label."
            )

st.markdown("---")


# ============================
# SECTION 2 — UPLOAD EXTERNAL FEATURE FILE
# ============================

st.header("📤 Upload External Feature File")

st.write(
    """
You can upload a **CSV file** containing exactly **one row of 24 DSP features**.  
This file should represent a new, unseen 30-second EEG epoch that is **not part of the training set**.
"""
)

uploaded_file = st.file_uploader("Upload .csv file with 24 features", type=["csv"])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file, header=None)
        if df_input.shape[1] != N_FEATURES:
            st.error(
                f"Expected {N_FEATURES} features, but the uploaded file has {df_input.shape[1]} columns."
            )
        else:
            x_uploaded = df_input.iloc[0].values.reshape(1, -1)
            st.success("File successfully loaded. Click the button below to run prediction.")

            if st.button("Run prediction on uploaded file"):
                y_id, y_label, probs = get_prediction(current_model, x_uploaded)

                st.subheader("Prediction (Uploaded File)")
                st.write(f"**Predicted class ID:** `{y_id}`")
                st.write(f"**Predicted sleep stage:** `{y_label}`")

                st.subheader("Class probabilities")
                df_probs = build_prob_dataframe(probs)
                st.bar_chart(df_probs)

                st.subheader("Conclusion")
                st.success(
                    f"The uploaded EEG feature vector is classified as **{y_label}**. "
                    "This demonstrates that our DSP + ML pipeline can generalize "
                    "to external inputs that were not part of the training set."
                )

    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")

st.markdown("---")
st.caption("Demo built for DSP501 – EEG Sleep Stage Classification via DSP + ML.")
