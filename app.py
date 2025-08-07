import streamlit as st
import numpy as np
import joblib

# Load the model, imputer, and scaler
model = joblib.load('Random_Forest_model.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')

st.title("❤️ Heart Disease Risk Assessment App")
st.write("Enter your health and lifestyle details to assess your risk of heart disease.")

# Input fields for the 15 features
st.subheader("Health and Lifestyle Information")
high_bp = st.selectbox("Do you have high blood pressure?", ["No", "Yes"], index=0)
high_chol = st.selectbox("Do you have high cholesterol?", ["No", "Yes"], index=0)
bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
smoker = st.selectbox("Are you a smoker?", ["No", "Yes"], index=0)
diabetes = st.selectbox("Do you have diabetes?", ["No", "Yes"], index=0)
phys_activity = st.selectbox("Do you engage in regular physical activity?", ["Yes", "No"], index=0)
fruits = st.selectbox("Do you consume fruits daily?", ["Yes", "No"], index=0)
veggies = st.selectbox("Do you consume vegetables daily?", ["Yes", "No"], index=0)
hvy_alcohol_consump = st.selectbox("Do you engage in heavy alcohol consumption?", ["No", "Yes"], index=0)
ment_hlth = st.number_input("Days of poor mental health (past 30 days)", min_value=0, max_value=30, value=0, step=1)
phys_hlth = st.number_input("Days of poor physical health (past 30 days)", min_value=0, max_value=30, value=0, step=1)
sex = st.selectbox("Sex", ["Female", "Male"], index=0)
age = st.slider("Age", min_value=18, max_value=100, value=50, step=1)
education = st.selectbox("Education Level",
                         ["Never attended", "Grades 1-8", "Grades 9-11", "High school graduate",
                          "Some college", "College graduate"], index=3)
income = st.selectbox("Annual Income Level",
                      ["Less than $10,000", "$10,000-$15,000", "$15,000-$20,000",
                       "$20,000-$25,000", "$25,000-$35,000", "$35,000-$50,000",
                       "$50,000-$75,000", "More than $75,000"], index=5)

# Convert inputs to appropriate values
high_bp_val = 1 if high_bp == "Yes" else 0
high_chol_val = 1 if high_chol == "Yes" else 0
smoker_val = 1 if smoker == "Yes" else 0
diabetes_val = 1 if diabetes == "Yes" else 0
phys_activity_val = 1 if phys_activity == "Yes" else 0
fruits_val = 1 if fruits == "Yes" else 0
veggies_val = 1 if veggies == "Yes" else 0
hvy_alcohol_consump_val = 1 if hvy_alcohol_consump == "Yes" else 0
sex_val = 1 if sex == "Male" else 0
education_val = ["Never attended", "Grades 1-8", "Grades 9-11",
                 "High school graduate", "Some college", "College graduate"].index(education) + 1
income_val = ["Less than $10,000", "$10,000-$15,000", "$15,000-$20,000",
              "$20,000-$25,000", "$25,000-$35,000", "$35,000-$50,000",
              "$50,000-$75,000", "More than $75,000"].index(income) + 1

# Collect input into array in the correct order
user_input = np.array([[high_bp_val, high_chol_val, bmi, smoker_val, diabetes_val,
                        phys_activity_val, fruits_val, veggies_val, hvy_alcohol_consump_val,
                        ment_hlth, phys_hlth, sex_val, age, education_val, income_val]])

# Predict
if st.button("Predict Risk"):
    user_input_imputed = imputer.transform(user_input)
    user_input_scaled = scaler.transform(user_input_imputed)
    prediction = model.predict(user_input_scaled)[0]
    probability = model.predict_proba(user_input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease! (Probability: {probability:.2%})")
        st.write("Please consult a healthcare professional for further evaluation.")
    else:
        st.success(f"✅ Low risk of heart disease. (Probability: {probability:.2%})")
        st.write("Continue maintaining a healthy lifestyle!")