import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor


# Load Dataset

dataset = pd.read_csv("Test.csv")

print("Dataset Shape:", dataset.shape)
print("\nColumns:")
print(dataset.columns)


# Fill Missing Values


# Fill Item_Weight
dataset["Item_Weight"] = dataset["Item_Weight"].fillna(
    dataset["Item_Weight"].mean()
)

# Fill Outlet_Size with mode of each Outlet_Type
mode = dataset.groupby("Outlet_Type")["Outlet_Size"].agg(
    lambda x: x.mode()[0]
)

dataset["Outlet_Size"] = dataset["Outlet_Size"].fillna(
    dataset["Outlet_Type"].map(mode)
)


# Standardize Item_Fat_Content


dataset.replace(
    {
        "Item_Fat_Content": {
            "low fat": "Low Fat",
            "LF": "Low Fat",
            "reg": "Regular"
        }
    },
    inplace=True
)


# Encode Categorical Columns


encoder = LabelEncoder()

categorical_columns = [
    "Item_Identifier",
    "Item_Fat_Content",
    "Item_Type",
    "Outlet_Identifier",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type"
]

for col in categorical_columns:
    dataset[col] = encoder.fit_transform(dataset[col])


# Features and Target
# Target = Item_MRP


X = dataset.drop(columns=["Item_MRP"])
y = dataset["Item_MRP"]


# Train-Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2
)

# Build Model

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=2
)


# Train Model


model.fit(X_train, y_train)


# Predictions

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Evaluation


train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

print("\n========== RESULTS ==========")
print(f"Training R² Score : {train_r2:.4f}")
print(f"Testing R² Score  : {test_r2:.4f}")
print(f"Training MAE      : {train_mae:.2f}")
print(f"Testing MAE       : {test_mae:.2f}")

# Sample Predictions

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": test_pred
})

print("\nFirst 10 Predictions:")
print(results.head(10))