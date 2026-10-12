import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from math import sqrt

data = pd.read_csv("diabetes.csv")

print(type(data))
# k co mac dinh 5
print(data.head(10))

print(data.info())
# explain
print(data.describe())
# hs tuong quan
# result = data.corr()
# dem tan so
print(data["Outcome"].value_counts())

# Draw figure
plt.figure(figsize=(8,8))
sn.histplot(data["Outcome"])
plt.title("Diabetes Distribution")
# plt.savefig("diabetes.jpg")
# plt.show()

# Set feature and Target
target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]

# Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
scaler = StandardScaler()

# result = scaler.fit_transform(x_train[["Pregnancies"]])
# for i, j in zip(x_train["Pregnancies"].values, result):
#     print("Before {} After {}".format(i, j))

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

cls = SVC()