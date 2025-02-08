import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model

# Load the trained model

model = load_model('regression_model_.h5')

# Load the encoders and scaler

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("ohe_encoder_geo.pkl", "rb") as file:
    ohe_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

## Streamlit app

st.title('Customer Salary estimation using Regression')

# User Input

geography = st.selectbox('Geography', ohe_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18,98)
Credit_Score = st.number_input('CreditScore')
tenure = st.slider('Tenure', 0,10)
balance = st.number_input('Balance')
No_of_products = st.slider('NumOfProducts', 1,5)
Has_credit_card = st.selectbox('HasCrCard',[0,1])
Is_Active_member = st.selectbox('IsActiveMember',[0,1])	
Exited = st.selectbox('Exited', [0,1])


# Prep input data

input_data = pd.DataFrame({
    'CreditScore': [Credit_Score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [No_of_products],
    'HasCrCard' : [Has_credit_card],
    'IsActiveMember': [Is_Active_member],
    'Exited' : [Exited]

})

## One hot encoding for 'Geography'

geo_encoder = ohe_encoder_geo.transform([[geography]]).toarray()
geo_encoder_df = pd.DataFrame(geo_encoder, columns=ohe_encoder_geo.get_feature_names_out(['Geography']))
                               
## Converting dict to df                     

input_df = pd.DataFrame(input_data)

## Combining OHE col with input data

input_df = pd.concat([input_df, geo_encoder_df], axis=1)

## Scaling the input data

input_data_scaled = scaler.transform(input_df)

# Prediction

prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]

print(f"Estimated salary of the user is , ${predicted_salary}")