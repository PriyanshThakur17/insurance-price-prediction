import pandas as pd
import numpy as np

print("Starting program...")

# Load dataset
df = pd.read_csv("insurance.csv")
print("Dataset Loaded Successfully!")

# Convert categorical to numeric
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

df = pd.get_dummies(df, columns=['region'], drop_first=True)

print("Data Preprocessing Done!")

# Split data
from sklearn.model_selection import train_test_split

X = df.drop('charges', axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("Model Trained!")

# Predict
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import mean_absolute_error, r2_score

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Example prediction
print("Sample Prediction:", model.predict(X_test[:1]))
import seaborn as sns
import matplotlib.pyplot as plt

# Age vs Charges
sns.scatterplot(x=df['age'], y=df['charges'])
plt.title("Age vs Insurance Charges")
plt.show()

# BMI vs Charges
sns.scatterplot(x=df['bmi'], y=df['charges'])
plt.title("BMI vs Insurance Charges")
plt.show()

# Smoker vs Charges (MOST IMPORTANT 🔥)
sns.boxplot(x=df['smoker'], y=df['charges'])
plt.title("Smoker vs Charges")
plt.show()