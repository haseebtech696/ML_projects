import pandas as pd
import numpy as np
from pygments.lexers import q
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("parkinsons.csv")
# print(dataset.head())
# print(dataset.isnull().sum())
# print(dataset["status"].value_counts())

# dataset.groupby("status").mean()

X = dataset.drop(columns=["name","status"])
y = dataset["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

model = StandardScaler()

model.fit(X_train)

X_train = model.transform(X_train)
X_text = model.transform(X_test)

model = svm.SVC(kernel="linear")
model.fit(X_train, y_train)
model.fit(X_train, y_train)

X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)

X_train_accuracy = accuracy_score(y_train,X_train_prediction)
X_test_accuracy = accuracy_score(y_test,X_test_prediction)

print(X_train_accuracy)
print(X_test_accuracy)
