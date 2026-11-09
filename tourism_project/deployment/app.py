import os
import json
import streamlit as st
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "best_tourism_package_model_v1.joblib")   # committed by the pipeline
CONFIG_PATH = os.path.join(BASE_DIR, "model_config.json")


@st.cache_resource
def load_model():
    # Load the model committed by the pipeline (sits next to this file)
    return joblib.load(MODEL_PATH)


def load_threshold(default=0.45):
    # Read the decision threshold selected during training (fallback to a default)
    try:
        with open(CONFIG_PATH) as f:
            return float(json.load(f)["classification_threshold"])
    except Exception:
        return default


st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="✈️", layout="centered")
st.title("Visit with Us - Wellness Tourism Package Prediction")
st.write("Fill in the customer details below to predict whether they will purchase the Wellness Tourism Package before contacting them.")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run the GitHub Actions pipeline so it commits the trained model.")
    st.stop()

model = load_model()
classification_threshold = load_threshold()

# ----------------------------
# Collect user input
# ----------------------------
st.subheader("Customer details")
col1, col2 = st.columns(2)
with col1:
    Age = st.slider("Age", 18, 70, 35)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Unmarried", "Divorced"])
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, max_value=200000.0, value=23000.0, step=500.0)
    CityTier = st.selectbox("City Tier", [1, 2, 3])
with col2:
    Passport = st.selectbox("Has Passport?", ["Yes", "No"])
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
    NumberOfTrips = st.slider("Average Number of Trips per Year", 1, 25, 3)
    NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 3)
    NumberOfChildrenVisiting = st.slider("Number of Children (below 5) Visiting", 0, 5, 1)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])

st.subheader("Interaction details")
col3, col4 = st.columns(2)
with col3:
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    DurationOfPitch = st.slider("Duration of Pitch (mins)", 0, 130, 15)
with col4:
    NumberOfFollowups = st.slider("Number of Follow-ups", 1, 10, 4)
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

# ----------------------------
# Save the inputs into a dataframe (same column names/types as training)
# ----------------------------
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

with st.expander("Show model input"):
    st.dataframe(input_data)

# Predict button
if st.button("Predict", type="primary"):
    prob = float(model.predict_proba(input_data)[0, 1])
    pred = int(prob >= classification_threshold)
    st.metric("Purchase probability", f"{prob:.1%}")
    if pred == 1:
        st.success("Prediction: the customer **is likely to purchase** the Wellness Tourism Package - prioritise this lead.")
    else:
        st.warning("Prediction: the customer **is unlikely to purchase** - consider a lower-cost campaign channel.")
    st.caption(f"Decision threshold: {classification_threshold}")
