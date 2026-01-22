# PRCP-1012: Game Winner Prediction (PUBG)

# -------------------------------
# Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Load Dataset
# -------------------------------
# Make sure CSV file is in same folder
df = pd.read_csv("PUBG_game_data.csv")

print("Dataset Loaded Successfully")
print(df.head())

# -------------------------------
# Basic Information
# -------------------------------
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------
# Data Cleaning
# -------------------------------
# Remove duplicate columns if any
df = df.loc[:, ~df.columns.duplicated()]

# Drop rows where target is missing
df = df.dropna(subset=["winPlacePerc"])

# Fill remaining missing values with 0
df = df.fillna(0)

# -------------------------------
# Encode Categorical Data
# -------------------------------
le = LabelEncoder()
df["matchType"] = le.fit_transform(df["matchType"])

# -------------------------------
# Feature Selection
# -------------------------------
X = df.drop(["winPlacePerc", "Id", "groupId", "matchId"], axis=1)
y = df["winPlacePerc"]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model 1: Linear Regression
# -------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_r2 = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Performance")
print("RMSE:", lr_rmse)
print("R2 Score:", lr_r2)

# -------------------------------
# Model 2: Random Forest Regressor
# -------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Performance")
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

# -------------------------------
# Model Comparison Report
# -------------------------------
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "RMSE": [lr_rmse, rf_rmse],
    "R2 Score": [lr_r2, rf_r2]
})

print("\nModel Comparison Report:")
print(comparison)

# -------------------------------
# Feature Importance (Random Forest)
# -------------------------------
feature_importance = pd.Series(
    rf.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# -------------------------------
# Visualization
# -------------------------------
plt.figure(figsize=(8,4))
sns.histplot(y, bins=20)
plt.title("Win Place Percentage Distribution")
plt.show()

# -------------------------------
# Conclusion
# -------------------------------
print("""
Conclusion:
Random Forest performed better than Linear Regression due to
its ability to handle non-linear relationships and feature interactions.
Hence, Random Forest is recommended for production use.
""")

# -------------------------------
# Challenges Faced
# -------------------------------
print("""
Challenges Faced:
1. Large number of features
2. Presence of categorical variable (matchType)
3. Non-linear data patterns
4. Missing values in dataset

Techniques Used:
- Label Encoding
- Feature Selection
- Random Forest Regressor
""")
