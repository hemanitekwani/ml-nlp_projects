


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import joblib
       

breast_data=pd.read_csv('/content/breast-cancer.csv')
    
breast_data.head()
     
breast_data.shape
     
breast_data.describe()
    
breast_data['diagnosis'].value_counts()
     
breast_data.groupby('diagnosis').mean()
     
X = breast_data.drop(columns='diagnosis', axis=1)
Y =  breast_data['diagnosis']
print(X)
print(Y)
         
breast_data.isnull().sum()
     
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
     

print(X.shape, X_train.shape, X_test.shape)
     

model = LogisticRegression()
model.fit(X_train, Y_train)
     
LogisticRegression()

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
     
print('Accuracy score of the training data : ', training_data_accuracy)
     
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
     
print('Accuracy score of the test data : ', test_data_accuracy)
     