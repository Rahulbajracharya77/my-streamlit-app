# ==========================================
# IMPORT LIBRARIES
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv(r'C:\Users\Ripple\Desktop\PROJECT\Car_Price_Prediction.csv')

# ==========================================
# DATA TYPES
# ==========================================
# Convert categorical columns
cat_cols = ['Make', 'Model', 'Fuel Type', 'Transmission']
df[cat_cols] = df[cat_cols].astype('category')

# Ensure numeric columns remain numeric
num_cols = ['Year', 'Engine Size', 'Mileage', 'Price']
df[num_cols] = df[num_cols].apply(pd.to_numeric)

# ==========================================
# EDA: PLOTS
# ==========================================

# Scatterplot: Year vs Price
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Year', y='Price', hue='Make', marker='*')
plt.title('Year vs Price by Make')
plt.show()

# Boxplot: Fuel Type vs Price
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x='Fuel Type', y='Price')
plt.title('Price Distribution by Fuel Type')
plt.show()

# Heatmap: correlation of numeric columns
plt.figure(figsize=(8,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# ==========================================
# FEATURE SELECTION
# ==========================================
# For simplicity, we’ll one-hot encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# PIPELINE: SCALING + REGRESSION
# ==========================================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)

# ==========================================
# PREDICTIONS
# ==========================================
y_pred = pipeline.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# ==========================================
# OPTIONAL: PREDICT FOR NEW DATA
# ==========================================
# Example: user provides a new car
# Make sure to match the one-hot columns
user_car = pd.DataFrame({
    'Year': [2020],
    'Engine Size': [2.0],
    'Mileage': [15000],
    'Make_Toyota': [1],      # set 1 for the correct make, 0 for others
    'Fuel Type_Diesel': [0],  # adjust as per your dataset
    'Fuel Type_Petrol': [1],
    'Transmission_Manual': [0],
    'Transmission_Automatic': [1]
})

# Add missing columns if needed
for col in X.columns:
    if col not in user_car.columns:
        user_car[col] = 0

user_pred_price = pipeline.predict(user_car[X.columns])
print(f"Predicted Price for the car: ${user_pred_price[0]:.2f}")
