import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

parkinson_data = pd.read_csv('/content/archive (16).zip')

X = parkinson_data.drop(['name','status'], axis=1)
Y = parkinson_data['status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_transform = scaler.transform(X_train)
X_test_transform = scaler.transform(X_test)

model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

X_train_predict = model.predict(X_train)
X_train_accuracy = accuracy_score(Y_train, X_train_predict)
print(X_train_accuracy)

X_test_predict = model.predict(X_test)
X_test_accuracy = accuracy_score(Y_test, X_test_predict)
print(X_test_accuracy)

input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

input_data_as_numpy = np.asarray(input_data)
reshaped_data = input_data_as_numpy.reshape(1,-1)

std_data = scaler.transform(reshaped_data)

prediction = model.predict(std_data)

if(prediction[0] == 1):
    print("person have Parkinson's disease")
else:
    print("person does not have Parkinson's disease")
