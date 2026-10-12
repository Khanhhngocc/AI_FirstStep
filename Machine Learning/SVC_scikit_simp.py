from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf1 = SVC(kernel='linear', C=3)
clf1.fit(X_train, y_train)

clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(X_train, y_train)

print(f'SVC: {clf1.score(X_test, y_test)}')
print(f'KNN: {clf2.score(X_test, y_test)}')


