import pandas as pd
import re
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('/content/Data (1).csv', encoding='ISO-8859-1')

df['Text'] = df.iloc[:, 2:27].apply(lambda row: ' '.join(row.dropna()), axis=1)

def preprocess_text(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z]", " ", text)
    return text

df['Text'] = df['Text'].apply(preprocess_text)

train, test = train_test_split(df, test_size=0.2, random_state=42)

countvector = CountVectorizer(ngram_range=(2, 2))
train_dataset = countvector.fit_transform(train['Text'])
test_dataset = countvector.transform(test['Text'])

randomclassifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
randomclassifier.fit(train_dataset, train['Label'])

api_key ='183ff1fc408b47af9eb93d6d671de3f5'
url = f'https://newsapi.org/v2/top-headlines?country=in&category=business&apiKey=183ff1fc408b47af9eb93d6d671de3f5'

response = requests.get(url)
if response.status_code == 200:
    news_data = response.json()
    if 'articles' in news_data:
        articles = news_data['articles']

        news_to_predict = []
        for article in articles:
            title = article['title']
            description = article['description']
            if description is not None:
             combined_text = title + " " + description
             news_to_predict.append(preprocess_text(combined_text))

        if len(news_to_predict) == 0:
         print("No articles found in the API response.")

        else:

         test_dataset = countvector.transform(news_to_predict)

         predictions = randomclassifier.predict(test_dataset)

         for i, article in enumerate(articles):
          if i < len(predictions):
            title = article['title']
            source = article['source']['name']
            publication_date = article['publishedAt']
            prediction = "Increase" if predictions[i] == 1 else "Decrease"
            print(f"Title: {title}")
            print(f"Source: {source}")
            print(f"Publication Date: {publication_date}")
            print(f"Prediction: {prediction}")
            print("-----------------------------")
    else:
        print("No articles found in the API response.")
else:
    print("Error fetching data from News API.")
     