import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

# ---------------- UI ---------------- #

st.title("💰 Insurance Price Predictor")

st.markdown("### Enter Details 👇")

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

# ---------------- Prediction ---------------- #

if st.button("Predict 💡"):
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

# ---------------- Graph Section ---------------- #

st.subheader("📊 Data Insights")

# Age vs Charges
fig1, ax1 = plt.subplots()
sns.scatterplot(x=df['age'], y=df['charges'], ax=ax1)
ax1.set_title("Age vs Insurance Charges")
st.pyplot(fig1)

# BMI vs Charges
fig2, ax2 = plt.subplots()
sns.scatterplot(x=df['bmi'], y=df['charges'], ax=ax2)
ax2.set_title("BMI vs Insurance Charges")
st.pyplot(fig2)

# Smoker vs Charges (IMPORTANT)
fig3, ax3 = plt.subplots()
sns.boxplot(x=df['smoker'], y=df['charges'], ax=ax3)
ax3.set_title("Smoker vs Charges")
st.pyplot(fig3)