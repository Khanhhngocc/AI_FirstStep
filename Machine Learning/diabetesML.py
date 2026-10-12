import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sn

data = pd.read_csv("diabetes.csv")

target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

cls = SVC()
cls.fit(x_train, y_train)

y_predict = cls.predict(x_test)

print(classification_report(y_test, y_predict))

# for i,j in zip(y_test, y_predict):
#     print("Actual {} Predict {}".format(i, j))