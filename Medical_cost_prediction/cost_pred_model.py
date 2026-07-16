import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('insurance.csv')
#
# print(dataset.isnull().sum())

sns.set()
plt.figure(figsize=(10,10))
sns.displot(dataset["age"])
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(10,10))
sns.countplot(x= "sex", data=dataset)
plt.title("Sex Distribution")
plt.show()

sns.set()
plt.figure(figsize=(10,10))
sns.displot(dataset["bmi"])
plt.title("BMI Distribution")
plt.show()

plt.figure(figsize=(10,10))
sns.countplot(x= "children", data=dataset)
plt.title("Children Distribution")
plt.show()

plt.figure(figsize=(10,10))
sns.countplot(x= "smoker", data=dataset)
plt.title("Smoker Distribution")
plt.show()

plt.figure(figsize=(10,10))
sns.countplot(x= "region", data=dataset)
plt.title("Region Distribution")
plt.show()

dataset.replace({"sex":{"male":0, "female":1}}, inplace=True)
dataset.replace({"smoker":{"yes":0, "no":1}}, inplace=True)
dataset.replace({"region":{"southeast":0, "southwest":1,"northeast":2,"northwest":3}}, inplace=True)
print(dataset.head())

X = dataset.drop(columns=["charges"])
Y = dataset["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

model = LinearRegression()
model.fit(X_train,y_train)

training_data_pred = model.predict(X_train)
testing_data_pred = model.predict(X_test)

r2_train = metrics.r2_score(y_train, training_data_pred)
r2_test = metrics.r2_score(y_test, testing_data_pred)

print(r2_train)
print(r2_test)


