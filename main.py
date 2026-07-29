import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler
model = joblib.load('xgboost_diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Diabetes Prediction Screening Tool")
st.markdown("Enter patient metrics below to assess diabetes risk.")

# Input fields for continuous variables
age = st.number_input("Age", min_value=0, max_value=100, value=40)
bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=10.0, value=5.5)
glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

# Input fields for categorical variables
hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
gender = st.selectbox("Gender", ["Female", "Male"])
smoking = st.selectbox("Smoking History", ["never", "current", "former", "ever", "not current"])

if st.button("Run Diagnostics"):
    # Create a DataFrame for the raw inputs
    input_df = pd.DataFrame({
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'bmi': [bmi],
        'HbA1c_level': [hba1c],
        'blood_glucose_level': [glucose],
        'gender': [gender],
        'smoking_history': [smoking]
    })

    # Replicate Feature Engineering
    bmi_bins = [0, 18.5, 24.9, 29.9, 100]
    bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    input_df['bmi_category'] = pd.cut(input_df['bmi'], bins=bmi_bins, labels=bmi_labels)

    age_bins = [0, 12, 19, 59, 150]
    age_labels = ['Child', 'Teen', 'Adult', 'Senior']
    input_df['age_group'] = pd.cut(input_df['age'], bins=age_bins, labels=age_labels)

    # One-hot encode inputs matching your training set process
    input_df = pd.get_dummies(input_df, columns=['gender', 'smoking_history', 'bmi_category', 'age_group'])

    # Define the exact expected columns from X_train
    expected_columns = [
        'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
        'gender_Male', 'smoking_history_current', 'smoking_history_ever', 
        'smoking_history_former', 'smoking_history_never', 'smoking_history_not current',
        'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese',
        'age_group_Teen', 'age_group_Adult', 'age_group_Senior'
    ]

    # Align columns (adds missing dummy columns as False/0, drops extra ones)
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Scale continuous features
    continuous_vars = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    input_df[continuous_vars] = scaler.transform(input_df[continuous_vars])

    # Predict using the weighted XGBoost model
    prediction = model.predict(input_df)
    
    # Output result
    if prediction[0] == 1:
        st.error("Diagnosis: High Risk (Diabetic)")
    else:
        st.success("Diagnosis: Low Risk (Healthy)")