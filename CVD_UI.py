#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open(r'C:\Users\keert\Downloads\boost_model_102.pkl', 'rb'))
scaler = pickle.load(open(r'C:\Users\keert\Downloads\scalar_CVD_102.pkl', 'rb'))

def predict_cardio_disease(age, gender, height, weight, ap_hi, cholesterol, gluc, smoke, alco, active):
    input_data = {'age': age,
                  'gender': gender,
                  'height': height,
                  'weight': weight,
                  'ap_hi': ap_hi,
                  'cholesterol': cholesterol,
                  'gluc': gluc,
                  'smoke': smoke,
                  'alco': alco,
                  'active': active}
    
    input_df = pd.DataFrame([input_data], columns = ['age','gender','height','weight','ap_hi','cholesterol','gluc','smoke','alco','active'])
    columns_to_scale = ['age', 'weight', 'ap_hi','height']
    input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])
    
    prediction = model.predict(input_df)
    
    return prediction[0]


st.title('Cardiovascular Disease Prediction')


age = st.number_input('Age', min_value=1, max_value=100, value=25)
gender = st.selectbox('Gender', ['Male', 'Female'])
height = st.number_input('Height (cm)', min_value=100, max_value=250, value=170)
weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70)
ap_hi = st.number_input('Systolic blood pressure', min_value=60, max_value=250, value=120)
cholesterol = st.selectbox('Cholesterol', ['Normal', 'Above normal', 'Well above normal'])
gluc = st.selectbox('Glucose', ['Normal', 'Above normal', 'Well above normal'])
smoke = st.selectbox('Smoking', ['No', 'Yes'])
alco = st.selectbox('Alcohol intake', ['No', 'Yes'])
active = st.selectbox('Physical activity', ['No', 'Yes'])


if gender == 'Male':
    gender = 2
else:
    gender = 1

if cholesterol == 'Normal':
    cholesterol = 1
elif cholesterol == 'Above normal':
    cholesterol = 2
else:
    cholesterol = 3
    
if gluc == 'Normal':
    gluc = 1
elif gluc == 'Above normal':
    gluc = 2
else:
    gluc = 3
    
if smoke == 'No':
    smoke = 0
else:
    smoke = 1
    
if alco == 'No':
    alco = 0
else:
    alco = 1
    
if active == 'No':
    active = 0
else:
    active = 1
    
age = age * 365



if st.button('Predict'):
    result = predict_cardio_disease(age, gender, height, weight, ap_hi, cholesterol, gluc, smoke, alco, active)
    if result == 0:
        st.subheader('You are not at risk of having cardiovascular disease :thumbsup:')
    else:
        st.subheader('You are at risk of having cardiovascular disease. :thumbsdown:')
        st.subheader('Please consult a doctor. 💊')


# In[ ]:




