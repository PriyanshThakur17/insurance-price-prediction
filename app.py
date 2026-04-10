import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load and preprocess data
df = pd.read_csv("insurance.csv")

df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['region'], drop_first=True)

X = df.drop('charges', axis=1)
y = df['charges']

model = LinearRegression()
model.fit(X, y)

# UI
st.title("💰 Insurance Price Predictor")

age = st.slider("Age", 18, 100)
sex = st.selectbox("Sex", ["Male", "Female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0)
children = st.number_input("Children", 0, 5)
smoker = st.selectbox("Smoker", ["Yes", "No"])
region = st.selectbox("Region", ["northwest", "southeast", "southwest"])

# Convert inputs
sex = 0 if sex == "Male" else 1
smoker = 1 if smoker == "Yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

if st.button("Predict"):
    input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region_northwest': [region_northwest],
    'region_southeast': [region_southeast],
    'region_southwest': [region_southwest]
})
    input_data = input_data[X.columns]
    prediction = model.predict(input_data)

    st.success(f"💸 Predicted Insurance Cost: ₹{int(prediction[0])}")