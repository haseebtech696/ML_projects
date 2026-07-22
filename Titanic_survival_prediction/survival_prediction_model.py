import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("train-selected-columns.csv")

dataset["Age"] = dataset["Age"].fillna(dataset["Age"].mean())
# print(dataset.head())
print(dataset.isnull().sum())

sns.set()
sns.countplot(x="Survived",data=dataset)
plt.show()

sns.countplot(x="Sex",data=dataset)
plt.show()

sns.countplot(x="Sex",hue="Survived",data=dataset)
plt.show()


dataset.replace({"Sex":{"male":0, "female":1}}, inplace=True)

X = dataset.drop(columns = ["Survived","PassengerId","Ticket","Name"])
Y = dataset["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)

model = LogisticRegression()
model.fit(X_train, y_train)

X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)

X_train_accuracy = accuracy_score(X_train_prediction, y_train)
X_test_accuracy = accuracy_score(X_test_prediction, y_test)

print(X_train_accuracy)
print(X_test_accuracy)