import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')
# print(stopwords.words('english'))

news_dataset = pd.read_csv("FakeNews.csv")
# print(news_dataset.shape)

# print(news_dataset.isnull().sum())
final_dataset = news_dataset.fillna('')

final_dataset['content'] = final_dataset['title'] + ' ' + final_dataset['author']

# print(final_dataset.isnull().sum())

# X = final_dataset.drop('label', axis=1)
# Y = final_dataset['label']

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

final_dataset['content']= final_dataset['content'].apply(stemming)

print(final_dataset.head())

X = final_dataset['content'].values
Y = final_dataset['label'].values

vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y,stratify=Y, test_size=0.2, random_state=2)

model = LogisticRegression()
model.fit(x_train, y_train)
