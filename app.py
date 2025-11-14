import streamlit as st
import pandas as pd, numpy as np, pickle, json
from pathlib import Path

BASE = Path(__file__).parent
model = pickle.load(open(BASE / "model.pkl", "rb"))
feature_columns = json.load(open(BASE / "feature_columns.json"))

st.set_page_config(page_title="Attrition Predictor", layout="centered")

st.title("Employee Attrition - Demo Streamlit App")

st.write("Fill employee details and click Predict. This demo model is synthetic and for illustration only.")

def user_inputs():
    data = {}
    # Numeric inputs
    data["Age"] = st.number_input("Age", min_value=18, max_value=70, value=30)
    data["DistanceFromHome"] = st.number_input("Distance From Home (km)", min_value=0, max_value=100, value=10)
    data["Education"] = st.selectbox("Education (1-5)", [1,2,3,4,5], index=2)
    data["MonthlyIncome"] = st.number_input("Monthly Income", min_value=0, max_value=100000, value=5000)
    data["NumCompaniesWorked"] = st.number_input("Num Companies Worked", min_value=0, max_value=50, value=2)
    data["TotalWorkingYears"] = st.number_input("Total Working Years", min_value=0, max_value=60, value=5)
    data["TrainingTimesLastYear"] = st.number_input("Training Times Last Year", min_value=0, max_value=20, value=2)
    data["YearsAtCompany"] = st.number_input("Years At Company", min_value=0, max_value=60, value=3)
    data["YearsSinceLastPromotion"] = st.number_input("Years Since Last Promotion", min_value=0, max_value=40, value=0)
    data["YearsWithCurrManager"] = st.number_input("Years With Current Manager", min_value=0, max_value=40, value=2)
    # Categorical inputs
    data["BusinessTravel"] = st.selectbox("Business Travel", ['Travel_Rarely','Travel_Frequently','Non-Travel'])
    data["Department"] = st.selectbox("Department", ['Sales','Research & Development','Human Resources'])
    data["EducationField"] = st.selectbox("Education Field", ['Life Sciences','Medical','Marketing','Technical Degree','Other'])
    data["Gender"] = st.selectbox("Gender", ['Male','Female'])
    data["JobRole"] = st.selectbox("Job Role", ['Sales Executive','Research Scientist','Laboratory Technician','Manufacturing Director','Healthcare Representative','Manager','Sales Representative','Research Director','Human Resources'])
    data["MaritalStatus"] = st.selectbox("Marital Status", ['Single','Married','Divorced'])
    data["OverTime"] = st.selectbox("OverTime", ['Yes','No'])
    return pd.DataFrame([data])

input_df = user_inputs()
st.subheader("Input preview")
st.write(input_df)

if st.button("Predict Attrition Probability"):
    proba = model.predict_proba(input_df)[:,1][0]
    pred = model.predict(input_df)[0]
    st.metric("Attrition probability", f"{proba:.3f}", delta=None)
    st.write("Prediction (1 = likely to leave, 0 = likely to stay):", int(pred))
    st.info("This is a demo model trained on synthetic data. For production use, train with your real dataset and validate thoroughly.")
