import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
dataset = pd.read_csv("winequality-red.csv")

# Display first 5 rows
print(dataset.head())

# Dataset information
print("\nDataset Shape:", dataset.shape)
print("\nMissing Values:")
print(dataset.isnull().sum())

# Count plot of quality
sns.catplot(x="quality", data=dataset, kind="count")
plt.show()

# Correlation matrix
correlation = dataset.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation,
    annot=True,
    cbar=True,
    square=True,
    fmt=".1f",
    annot_kws={"size": 8},
    cmap="Blues"
)
plt.show()

# Features (X) and Target (Y)
X = dataset.drop("quality", axis=1)

# Convert quality into binary classes
# Good wine (quality >= 7) -> 1
# Bad/Average wine (quality < 7) -> 0
Y = dataset["quality"].apply(lambda y: 1 if y >= 7 else 0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=3,
    stratify=Y
)

# Create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=3
)

model.fit(X_train, y_train)

# Predictions
X_train_pred = model.predict(X_train)
X_test_pred = model.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, X_train_pred)
test_acc = accuracy_score(y_test, X_test_pred)

print("\nTraining Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

