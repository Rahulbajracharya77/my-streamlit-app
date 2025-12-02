import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# STREAMLIT PAGE TITLE
# ==========================================
st.title(" Car Price Prediction App")
st.write(
    "This app performs EDA and builds a Linear Regression model to predict car prices."
)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(r"C:\Users\Ripple\Desktop\PROJECT\PROJECT 2/Car_Price_Prediction.csv")
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ==========================================
# DATA TYPES
# ==========================================
cat_cols = ["Make", "Model", "Fuel Type", "Transmission"]
num_cols = ["Year", "Engine Size", "Mileage", "Price"]

df[cat_cols] = df[cat_cols].astype("category")
df[num_cols] = df[num_cols].apply(pd.to_numeric)

# ==========================================
# EDA: PLOTS
# ==========================================
st.subheader("Exploratory Data Analysis")

# Scatterplot
st.write("### Year vs Price by Make")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x="Year", y="Price", hue="Make", ax=ax1)
ax1.set_xticklabels(ax1.get_xticks(), rotation=45)
plt.tight_layout()
st.pyplot(fig1)

# Boxplot
st.write("### Price Distribution by Fuel Type")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x="Fuel Type", y="Price", ax=ax2)
ax2.set_xticklabels(ax2.get_xticks(), rotation=45)
plt.tight_layout()
st.pyplot(fig2)

# Heatmap
st.write("### Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax3)
plt.tight_layout()
st.pyplot(fig3)

# ==========================================
# FEATURE ENGINEERING
# ==========================================
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Price", axis=1)
y = df_encoded["Price"]

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# PIPELINE: SCALER + LINEAR REGRESSION
# ==========================================
pipeline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
pipeline.fit(X_train, y_train)

# ==========================================
# MODEL PERFORMANCE OUTPUT
# ==========================================
st.subheader("Model Performance")
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**R2 Score:** {r2:.2f}")

# ==========================================
# SAFE NEW CAR PREDICTION
# ==========================================
st.subheader("Predict Price for a New Car")

new_car = {
    "Year": 2020,
    "Engine Size": 2.0,
    "Mileage": 15000,
    "Make": "Toyota",
    "Model": "Corolla",
    "Fuel Type": "Petrol",
    "Transmission": "Automatic",
}

new_df = pd.DataFrame([new_car])

# One-hot encode new input
new_df_encoded = pd.get_dummies(new_df, drop_first=True)

# Match training columns
for col in X.columns:
    if col not in new_df_encoded.columns:
        new_df_encoded[col] = 0

new_df_encoded = new_df_encoded[X.columns]

predicted_price = pipeline.predict(new_df_encoded)[0]

st.write(f"### Predicted Price: **${predicted_price:.2f}**")

