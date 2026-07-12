import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


dataset = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')


print("Missing Values:\n")
print(dataset.isnull().sum())


X = dataset.drop(columns=['selling_price'])
y = dataset['selling_price']


X = pd.get_dummies(X, drop_first=True)

print("\nShape after encoding:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=2
)


model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


train_pred = model.predict(X_train)
test_pred = model.predict(X_test)


train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))


print(f"Train R² : {train_r2:.4f}")
print(f"Test R²  : {test_r2:.4f}")
print(f"Train MAE: {train_mae:.2f}")
print(f"Test MAE : {test_mae:.2f}")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE : {test_rmse:.2f}")

