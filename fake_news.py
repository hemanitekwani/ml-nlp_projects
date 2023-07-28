

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

df = pd.read_csv('/content/train (1).csv', encoding='latin', nrows=1000)
df.head()
df.shape
df.isnull().sum()

news_dataset = df.fillna('')
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

X = news_dataset['content']
Y = news_dataset['label']

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

X = news_dataset['content'].values
Y = news_dataset['label'].values

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)

train_accuracy = model.score(X_train, Y_train)
print('Train Accuracy:', train_accuracy)

test_accuracy = model.score(X_test, Y_test)
print('Test Accuracy:', test_accuracy)

X_new = X_test[0]
X_new = X_new.reshape(1, -1)
prediction = model.predict(X_new)

if prediction[0] == 0:
    print('The news is Real')
else:
    print('The news is Fake')
