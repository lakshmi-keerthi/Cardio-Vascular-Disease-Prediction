import streamlit as st
import pandas as pd
import pickle

with open('boost_model_102.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scalar_CVD_102.pkl', 'wb') as f:
    pickle.dump(scaler, f)

def predict_cardio_disease(age, gender, height, weight, ap_hi, cholesterol, gluc, smoke, alco, active):
    input_data = {
        'age': age,
        'gender': gender,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': smoke,
        'alco': alco,
        'active': active
    }

    df = pd.DataFrame([input_data])
    cols_to_scale = ['age', 'weight', 'ap_hi', 'height']
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return model.predict(df)[0]

st.title('🫀 Cardiovascular Disease Prediction')

# User Inputs
age = st.slider('Age (in years)', 1, 100, 25)
gender = st.selectbox('Gender', ['Male', 'Female'])
height = st.slider('Height (cm)', 100, 250, 170)
weight = st.slider('Weight (kg)', 30, 200, 70)
ap_hi = st.slider('Systolic Blood Pressure', 60, 250, 120)
cholesterol = st.selectbox('Cholesterol', ['Normal', 'Above normal', 'Well above normal'])
gluc = st.selectbox('Glucose', ['Normal', 'Above normal', 'Well above normal'])
smoke = st.selectbox('Smoking', ['No', 'Yes'])
alco = st.selectbox('Alcohol Intake', ['No', 'Yes'])
active = st.selectbox('Physical Activity', ['No', 'Yes'])

# Encoding
gender = 2 if gender == 'Male' else 1
cholesterol = ['Normal', 'Above normal', 'Well above normal'].index(cholesterol) + 1
gluc = ['Normal', 'Above normal', 'Well above normal'].index(gluc) + 1
smoke = 1 if smoke == 'Yes' else 0
alco = 1 if alco == 'Yes' else 0
active = 1 if active == 'Yes' else 0
age_days = age * 365

if st.button('Predict'):
    result = predict_cardio_disease(age_days, gender, height, weight, ap_hi, cholesterol, gluc, smoke, alco, active)
    if result == 0:
        st.success('✅ You are not at risk of cardiovascular disease.')
    else:
        st.error('⚠️ You are at risk of cardiovascular disease. Please consult a doctor.')
