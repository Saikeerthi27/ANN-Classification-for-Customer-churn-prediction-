import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle
from tensorflow.keras.models import load_model

# Load the trained model

model = load_model('model.h5')

# Load the encoders and scaler

with open('label_encoder_gender.pkl','rb') as file:
   label_encoder_gender = pickle.load(file)

with open('ohe_encoder_geo.pkl','rb') as file:
    ohe_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


## Streamlit app

st.title('Customer churn analysis app')

## User Input

geography = st.selectbox('Geography', ohe_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18,98)
Credit_Score = st.number_input('CreditScore')
tenure = st.slider('Tenure', 0,10)
balance = st.number_input('Balance')
No_of_products = st.slider('NumOfProducts', 1,5)
Has_credit_card = st.selectbox('HasCrCard',[0,1])
Is_Active_member = st.selectbox('IsActiveMember',[0,1])	
Estimated_Salary = st.number_input('EstimatedSalary')

## Prepare the input data

input_data = ({
    'CreditScore': [Credit_Score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [No_of_products],
    'HasCrCard' : [Has_credit_card],
    'IsActiveMember': [Is_Active_member],
    'EstimatedSalary': [Estimated_Salary]
})

## One hot encoding for 'Geography'

geo_encoder = ohe_encoder_geo.transform([[geography]]).toarray()
geo_encoder_df = pd.DataFrame(geo_encoder, columns=ohe_encoder_geo.get_feature_names_out(['Geography']))
                               
## Converting dict to                               

input_df = pd.DataFrame(input_data)

## Combining OHE col with input data

input_df = pd.concat([input_df, geo_encoder_df], axis=1)

## Scaling the input data

input_data_scaled = scaler.transform(input_df)

# Predict churn

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba >= 0.5:
    st.write(f"Customer is likely to churn with {prediction_proba*100:.2f}% ")
else:
    st.write(f"Customer is not likely to churn because it just has {prediction_proba*100:.2f}%")



