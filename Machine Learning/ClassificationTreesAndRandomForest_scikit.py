from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

clfRf = RandomForestClassifier()
clfRf.fit(X_train, y_train)

print(f'DTC: {clf.score(X_test, y_test)}')
print(f'RFC: {clfRf.score(X_test, y_test)}')