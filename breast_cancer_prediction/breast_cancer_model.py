import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset = sklearn.datasets.load_breast_cancer()
# print(dataset)
dataset_frame = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# print(dataset_frame.head())

dataset_frame["label"] = dataset.target
# print(dataset_frame.isnull().sum())
dataset_frame.groupby("label").mean()

X = dataset_frame.drop("label", axis=1)
Y = dataset_frame["label"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)

X_test_score = accuracy_score(X_test_prediction, Y_test)
X_train_score = accuracy_score(X_train_prediction, Y_train)
print(X_test_score)
print(X_train_score)