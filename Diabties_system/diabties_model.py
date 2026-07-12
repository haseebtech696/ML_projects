import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

dataset= pd.read_csv("diabetes.csv")
# print(dataset.head())
# print(dataset.shape)
# print(dataset.describe())
# print(dataset.isnull().sum())
X = dataset.drop(["Outcome"],axis=1)
Y = dataset["Outcome"]
scalar = StandardScaler()
scalar.fit(X)
standerized_data= scalar.transform(X)
# print(standerized_data)
X=  standerized_data
y = dataset["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,stratify=y, random_state=2)

classifier= svm.SVC(kernel="linear")
classifier.fit(X_train,y_train)

X_train_prediction=classifier.predict(X_train)
train_score= accuracy_score(X_train_prediction,y_train)
# print(train_score)

X_test_prediction=classifier.predict(X_test)
test_score= accuracy_score(X_test_prediction,y_test)
# print(test_score)

input_data=(2,88,58,26,16,28.4,0.766,22)

data_in_array= np.array(input_data)
array_reshape=np.reshape(data_in_array,(1,-1))

std_scalar=scalar.transform(array_reshape)
prediction = classifier.predict(std_scalar)
print(prediction)
