import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("heart_disease_data.csv")

X = dataset.drop(["target"], axis = 1)
Y = dataset["target"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,stratify=Y, random_state=2)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)
X_train_accuracy = accuracy_score(X_train_prediction, y_train)
X_test_accuracy = accuracy_score(X_test_prediction, y_test)
print(X_train_accuracy)
print(X_test_accuracy)

#Logistic Regression works more better if it is standarized first and then aplly logisstic regression
#Difference between linear regression and logistic regression is that in linear regression we have to deal with numbers while in logistic regression we have to deal with category calues like true or false