import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

dataset = pd.read_csv('gld_price_data.csv')
# print(dataset.head())

# print(dataset.isnull().sum())
dataset = dataset.drop('Date', axis=1)

correlation = dataset.corr()

plt.figure(figsize = (10,10))
sns.heatmap(correlation, annot=True, fmt=".1f", cmap="Blues", cbar=True,annot_kws={"size": 6})
# plt.show()

X = dataset.drop(["GLD"],axis = 1)
Y= dataset["GLD"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=22)

regressor = RandomForestRegressor(n_estimators=100)

regressor.fit(X_train, y_train)
test_predictions = regressor.predict(X_test)
error_score = metrics.r2_score(y_test, test_predictions)
print(error_score)