import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load dataset
dataset = pd.read_csv("Loan_data.csv")

# Remove missing values
f_dataset = dataset.dropna().copy()

# Convert categorical columns to numeric
f_dataset = f_dataset.replace({
    "Loan_Status": {"Y": 1, "N": 0},
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Dependents": {"3+": 4},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2}
})

# Convert columns from object to numeric
cols = [
    "Loan_Status",
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

f_dataset[cols] = f_dataset[cols].astype(int)

# Optional: Visualize Education vs Loan Status
# sns.countplot(x="Education", hue="Loan_Status", data=f_dataset)
# plt.show()

# Features and target
X = f_dataset.drop(columns=["Loan_ID", "Loan_Status"])
y = f_dataset["Loan_Status"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    stratify=y,
    random_state=2
)

# Create and train the SVM model
classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, y_train)

# Training accuracy
X_train_pred = classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, X_train_pred)
print("Training Accuracy:", train_accuracy)

# Testing accuracy
X_test_pred = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, X_test_pred)
print("Testing Accuracy:", test_accuracy)