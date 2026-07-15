import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv("creditcard.csv")

legal = dataset[dataset["Class"] == 0]
fraud = dataset[dataset["Class"] == 1]

legal_sample = legal.sample(n=492, random_state=2)
new_dataset = pd.concat([legal_sample, fraud], axis=0)
scaler = StandardScaler()

X = new_dataset.drop("Class", axis=1)
y = new_dataset["Class"]

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=2
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Accuracy
train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
test_accuracy = metrics.accuracy_score(y_test, y_test_pred)

print("Accuracy on training set: {:.3f}".format(train_accuracy))
print("Accuracy on testing set: {:.3f}".format(test_accuracy))