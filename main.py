"""
Streamlit Crop Yield Prediction Prototype
File: streamlit_crop_yield_prototype.py

How to run:
1. Create a virtualenv and activate it.
2. pip install -r requirements.txt
3. streamlit run streamlit_crop_yield_prototype.py

Simplified version:
- Always uses generated sample dataset (no CSV upload).
- No data preview shown.
- Train model, predict manually, get recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import io
import base64

# ---------------------- Localization ----------------------
LANGS = {
    "en": {
        "title": "Crop Yield Prediction (Prototype)",
        "train_button": "Train Model",
        "predict_button": "Predict Yield",
        "model_trained": "Model trained! Metrics:",
        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R²",
        "download_model": "Download model",
        "input_form": "Manual input for prediction",
        "recommendation": "Recommendations",
        "fert_reco": "Fertilizer suggestion (N kg/ha)",
        "irr_reco": "Irrigation suggestion (times during season)",
        "notes": "Notes / How to use",
    },
    "hi": {
        "title": "फसल उपज भविष्यवाणी (प्रोटोटाइप)",
        "train_button": "मॉडल प्रशिक्षित करें",
        "predict_button": "उपज अनुमान",
        "model_trained": "मॉडल प्रशिक्षित हुआ! मेट्रिक्स:",
        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R²",
        "download_model": "मॉडल डाउनलोड करें",
        "input_form": "हाथ द्वारा इनपुट (पूर्वानुमान के लिए)",
        "recommendation": "सिफारिशें",
        "fert_reco": "उर्वरक सुझाव (N किग्रा/हेक्टेयर)",
        "irr_reco": "सिंचाई सुझाव (सीज़न के दौरान बार)",
        "notes": "नोट्स / उपयोग कैसे करें",
    }
}

# ---------------------- Utility functions ----------------------
def generate_sample_data(n=800, random_state=42):
    rng = np.random.RandomState(random_state)
    data = pd.DataFrame()
    crops = ["wheat", "rice", "maize"]
    data["crop"] = rng.choice(crops, size=n, p=[0.4, 0.35, 0.25])
    data["avg_temp"] = rng.normal(25, 4, size=n)
    data["total_rain"] = rng.normal(300, 120, size=n)
    data["soil_pH"] = rng.normal(6.5, 0.6, size=n)
    data["soil_N"] = rng.normal(200, 80, size=n)
    data["soil_P"] = rng.normal(30, 10, size=n)
    data["soil_K"] = rng.normal(150, 60, size=n)
    data["fert_N_applied"] = (data["soil_N"] * 0.1 + rng.normal(50, 25, size=n)).clip(0, 400)
    data["irrigation_events"] = rng.poisson(5, size=n)
    base = np.where(data["crop"] == "wheat", 3000,
                    np.where(data["crop"] == "rice", 4500, 3500))
    temp_effect = -20 * (data["avg_temp"] - 25)
    rain_effect = 0.8 * (data["total_rain"] - 300)
    fertility_effect = 1.2 * (data["soil_N"] + 0.5*data["soil_P"] + 0.3*data["soil_K"]) / 100
    fert_applied_effect = 1.0 * (data["fert_N_applied"]) / 50
    irrigation_effect = 50 * (data["irrigation_events"])
    noise = rng.normal(0, 250, size=n)
    data["yield"] = (base + temp_effect + rain_effect + fertility_effect +
                     fert_applied_effect + irrigation_effect + noise).clip(200, 10000)
    return data

def preprocess(df):
    df = df[["crop","avg_temp","total_rain","soil_pH","soil_N","soil_P","soil_K",
             "fert_N_applied","irrigation_events","yield"]].dropna()
    df = pd.get_dummies(df, columns=["crop"], drop_first=True)
    return df

def train_model(X, y, random_state=42):
    model = RandomForestRegressor(n_estimators=200, max_depth=12,
                                  random_state=random_state, n_jobs=-1)
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)   # old sklearn doesn’t support squared arg
    rmse = mse ** 0.5                         # manually compute RMSE
    r2 = r2_score(y_test, preds)
    return {"mae": mae, "rmse": rmse, "r2": r2}, preds

def recommend_actions(row):
    fert_suggest = 0
    if row.get("soil_N", 0) < 150:
        fert_suggest = max(0, (150 - row.get("soil_N", 0)) / 2)
    irr_suggest = 0
    if row.get("total_rain", 0) < 250:
        irr_suggest = 2
    return {
        "fert_N_additional_kg_per_ha": round(fert_suggest, 1),
        "irrigation_additional_events": irr_suggest
    }

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Crop Yield Prototype", layout="wide")

lang_choice = st.sidebar.selectbox("Language / भाषा", ["en","hi"],
                                   format_func=lambda x: "English" if x=="en" else "हिन्दी")
L = LANGS[lang_choice]

st.title(L["title"])
st.markdown("---")

# Always use sample dataset
df = generate_sample_data(n=800)

# Training
st.header(L["train_button"])
with st.expander("Training settings"):
    test_size = st.slider("Test size (%)", 10, 40, 20)
    random_state = st.number_input("Random seed", value=42)

if st.button(L["train_button"]):
    df_proc = preprocess(df)
    X = df_proc.drop(columns=["yield"])
    y = df_proc["yield"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=random_state)
    with st.spinner("Training model..."):
        model = train_model(X_train, y_train, random_state)
    metrics, preds = evaluate_model(model, X_test, y_test)
    st.success(L["model_trained"])
    st.write(f"{L['mae']}: {metrics['mae']:.1f}")
    st.write(f"{L['rmse']}: {metrics['rmse']:.1f}")
    st.write(f"{L['r2']}: {metrics['r2']:.3f}")

    # Save model for download
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="yield_model.joblib">{L["download_model"]}</a>'
    st.markdown(href, unsafe_allow_html=True)

    st.session_state["model"] = model
    st.session_state["model_features"] = X.columns.tolist()

# Prediction form
st.markdown("---")
st.header(L["input_form"])
with st.form(key="predict_form"):
    crop = st.selectbox("Crop", ["wheat","rice","maize"])
    avg_temp = st.number_input("Average temperature (°C)", value=25.0)
    total_rain = st.number_input("Total rainfall (mm)", value=300.0)
    soil_pH = st.number_input("Soil pH", value=6.5)
    soil_N = st.number_input("Soil Nitrogen (kg/ha)", value=150.0)
    soil_P = st.number_input("Soil Phosphorus (kg/ha)", value=30.0)
    soil_K = st.number_input("Soil Potassium (kg/ha)", value=150.0)
    fert_N_applied = st.number_input("Fertilizer N applied (kg/ha)", value=50.0)
    irrigation_events = st.number_input("Irrigation events", value=3)
    submit_predict = st.form_submit_button(L["predict_button"])

if submit_predict:
    if "model" not in st.session_state:
        st.warning("Please train a model first.")
    else:
        model = st.session_state["model"]
        features = {
            "avg_temp": avg_temp,
            "total_rain": total_rain,
            "soil_pH": soil_pH,
            "soil_N": soil_N,
            "soil_P": soil_P,
            "soil_K": soil_K,
            "fert_N_applied": fert_N_applied,
            "irrigation_events": irrigation_events,
            "crop_maize": 1 if crop=="maize" else 0,
            "crop_rice": 1 if crop=="rice" else 0,
        }
        X_pred = pd.DataFrame([features])
        X_pred = X_pred.reindex(columns=st.session_state["model_features"], fill_value=0)
        pred = model.predict(X_pred)[0]
        st.metric("Predicted yield (kg/ha)", f"{pred:.1f}")

        rec = recommend_actions({**features, "total_rain": total_rain})
        st.subheader(L["recommendation"])
        st.write(f"{L['fert_reco']}: {rec['fert_N_additional_kg_per_ha']} kg/ha")
        st.write(f"{L['irr_reco']}: {rec['irrigation_additional_events']} additional events")


