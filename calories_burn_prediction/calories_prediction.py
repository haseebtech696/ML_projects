import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


calories_data = pd.read_csv('calories.csv')
exercise_data = pd.read_csv('exercise.csv')
# print(calories_data.head())
# print(exercise_data.head())


final_dataset = pd.concat([exercise_data,calories_data["Calories"]], axis=1)
# print(final_dataset.isnull().sum())


sns.set()

sns.countplot(final_dataset["Gender"])
plt.show()

sns.distplot((final_dataset["Age"]))
plt.show()

sns.distplot((final_dataset["Height"]))
plt.show()

sns.distplot((final_dataset["Weight"]))
plt.show()

correlation = final_dataset.select_dtypes(include=np.number).corr()

plt.figure(figsize=(10,10))
sns.heatmap(correlation,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            square=True,
            cbar=True)
plt.show()


final_dataset.replace({"Gender":{"male":0, "female":1}}, inplace=True)

X = final_dataset.drop(columns = ["Calories","User_ID"])
Y = final_dataset["Calories"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
model = XGBRegressor()
model.fit(X_train, y_train)

test_data_predict = model.predict(X_test)
mae = metrics.mean_absolute_error(y_test, test_data_predict)
print(mae)