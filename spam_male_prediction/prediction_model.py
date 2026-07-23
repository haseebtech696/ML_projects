import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
dataset = pd.read_csv("mail_data.csv")

# Replace missing values with empty strings
mail_data = dataset.fillna("")

# Convert labels to numerical values
mail_data["Category"] = mail_data["Category"].map({
    "spam": 0,
    "ham": 1
})

# Features and Labels
X = mail_data["Message"]
Y = mail_data["Category"]

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2
)

# Convert text into TF-IDF features
feature_extraction = TfidfVectorizer(
    min_df=1,
    stop_words="english",
    lowercase=True
)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_features, Y_train)

# Training Accuracy
training_prediction = model.predict(X_train_features)
training_accuracy = accuracy_score(Y_train, training_prediction)
print("Training Accuracy:", training_accuracy)

# Testing Accuracy
testing_prediction = model.predict(X_test_features)
testing_accuracy = accuracy_score(Y_test, testing_prediction)
print("Testing Accuracy:", testing_accuracy)

input_mail = [
    "Congratulations! You have won a free iPhone. Click here to claim your prize."
]

# Convert input to features
input_data_features = feature_extraction.transform(input_mail)

# Make prediction
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print("Ham Mail")
else:
    print("Spam Mail")