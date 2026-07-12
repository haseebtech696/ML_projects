import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

price_dataset = fetch_california_housing()
# print(price_dataset)

final_dataset = pd.DataFrame(price_dataset.data, columns =price_dataset.feature_names)
# print(final_dataset.head())

final_dataset["price"]=price_dataset.target
# print(final_dataset.head())

# print(final_dataset.isnull().sum())

corelation = final_dataset.corr().corr()
plt.figure(figsize = (10,10))
sns.heatmap(corelation,cbar = True,annot=True,fmt=".1f",square=True,annot_kws={"size":10},cmap="Blues")
# plt.show()

X = final_dataset.drop(["price"],axis=1)
Y = final_dataset["price"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

model = XGBRegressor()
model.fit(X_train,y_train)
training_data_pred = model.predict(X_train)
# print(training_data_pred)

score_1 = metrics.r2_score(y_train,training_data_pred)
score_2 = metrics.mean_squared_error(y_train,training_data_pred)

# print(score_1)
# print(score_2)


