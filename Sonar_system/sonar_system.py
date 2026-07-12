import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv("Copy of sonar data.csv",header=None)
# print(solar_data.head())

# print(solar_data.describe())

# print(sonar_data.groupby(60).mean())

X = sonar_data.drop(60, axis=1)
y = sonar_data[60]
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.1,stratify = y, random_state=1)

model = LogisticRegression()

model.fit(X_train, y_train)

X_train_prediction=model.predict(X_train)
accuracy_score2=accuracy_score(X_train_prediction,y_train)

# print(accuracy_score)

X_test_prediction=model.predict(X_test)
accuracy_score1=accuracy_score(X_test_prediction,y_test)
# print(accuracy_score1)

input_data=(0.009, 0.0062, 0.0253, 0.0489, 0.1197, 0.1589, 0.1392, 0.0987, 0.0955, 0.1895, 0.1896, 0.2547, 0.4073, 0.2988, 0.2901, 0.5326, 0.4022, 0.1571, 0.3024, 0.3907, 0.3542, 0.4438, 0.6414, 0.4601, 0.6009, 0.869, 0.8345, 0.7669, 0.5081, 0.462, 0.538, 0.5375, 0.3844, 0.3601, 0.7402, 0.7761, 0.3858, 0.0667, 0.3684, 0.6114, 0.351, 0.2312, 0.2195, 0.3051, 0.1937, 0.157, 0.0479, 0.0538, 0.0146, 0.0068, 0.0187, 0.0059, 0.0095, 0.0194, 0.008, 0.0152, 0.0158, 0.0053, 0.0189, 0.0102)
input_data_array = np.asarray(input_data)
input_data_reshaped = input_data_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)