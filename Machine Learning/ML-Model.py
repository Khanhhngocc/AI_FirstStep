from scipy.cluster.vq import kmeans
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df1 = load_breast_cancer(as_frame=True).frame
#Same*
# data = load_breast_cancer()
# X = data.data
# y = data.target
# import, export in pandas data frame --> as_frame=True

print(df1)

df2 = fetch_california_housing(as_frame=True).frame
print(df2)

df2.hist(figsize=(14, 8))
plt.tight_layout()

from sklearn.datasets import make_blobs #generate randomly data according to certain distributions or structures
#make blobs for example as the name says creates clusters, cluster of data points

X, y = make_blobs(n_samples=5000, centers=5) #having 5000 instances and 5 blob centers (5 clusters), n_features features
plt.scatter(X[:, 0], X[:, 1], c=y) #take all instances with first dimension and take all instances with second dimension, c=y color to the labels

from sklearn.datasets import make_moons
X, y = make_moons(noise=0.2, random_state=1) #don't give random-state --> change everytime
plt.scatter(X[:, 0], X[:, 1], c=y)





#SPLITING DATA
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


import numpy as np
import matplotlib.pyplot as plt

counts = np.bincount(y_train)
positions = np.arange(3)
plt.bar(positions, counts)
plt.xticks(positions, data.target_names)

counts = np.bincount(y_test)
positions = np.arange(3)
plt.bar(positions, counts)
plt.xticks(positions, data.target_names)
#chat gpt

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2) #we only want single split,

for train_idx, test_idx in split.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

counts = np.bincount(y_test)
positions = np.arange(3)
plt.bar(positions, counts)
plt.xticks(positions, data.target_names)










#PREPROCESSING: is that you get some data, and the data is not perfect to be used right away with your model
#some of the feature might not even be numerical, some feature might not be scaled yet, so we processing the data before
# using for training and evaluating the model


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
#scaler have to learn certain value, take mean, variance of sample, do calculation

X_test_scaled = scaler.transform(X_test)
#use the same mean and variance before --> transform no fit

# bring to unit variance
(X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)  #is this a second method for scaling? if not then why need to do this???

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() #diferent scale, this don't have negative value, value between 0 - 1

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_min = np.min(X_train, axis=0)
X_max = np.max(X_train, axis=0)

(X_train - X_min) / (X_max - X_min)



#FEATURE ENCODING
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder

data = fetch_openml('car', as_frame=True).frame #this data cannot feed into, all ML algorithms work with numbers

columns_to_encode = ['lug_boot', 'safety']

encoder = OrdinalEncoder(
    categories=[
        ['small', 'med', 'big'],
        ['low', 'med', 'high']
    ]
)

data[columns_to_encode] = encoder.fit_transform(data[columns_to_encode])

#inverse
encoder.inverse_transform(data[columns_to_encode])


#one hot coding
data = fetch_openml('adult', as_frame=True).frame

from sklearn.preprocessing import OneHotEncoder
data.occupation.value_counts()

import pandas as pd
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
#get value later on, so was not part of the training data --> ignore it, don't handle it
#get individual column

encoded_values = encoder.fit_transform(data[('occupation', 'race')]) #get value but also need column values or name
new_cols = encoder.get_feature_names_out(['occupation', 'race'])

df_encoded = pd.DataFrame(encoded_values, columns=new_cols, index=data.index) #original dataframe have the same index
data_final = pd.concat(
    [data.drop(column=['occupation', 'race']), df_encoded],
    axis=1
)




#CLASSIFICATION

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

#How well the model perform in the testing data
knn.score(X_test_scaled, y_test)

single_instance = X_test_scaled[0]

# Predict
knn.predict([single_instance])

#Check
print(y_test[0])

knn.predict_proba([single_instance])



from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
#...


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
#...


from sklearn.svm import SVC
svm = SVC() #SVC doesn't have predict_proba()


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
#...


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
#...




#REGRESSION
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
reg = LinearRegression()

#fit
reg.fit(X_train_scaled, y_train)

#score
reg.score(X_test_scaled, y_test) # R squared - Score

single_instance = X_test_scaled[0]
print(single_instance)

reg.predict([single_instance])


from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor



#CLUSTERING
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=5000, centers=5, random_state=10)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1])


kmeans = KMeans(n_clusters=5) #looking for 5 cluster
kmeans.fit(X_scaled)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_)



from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.2)
dbscan.fit(X_scaled)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan.labels_)





#PCA
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = fetch_openml('mnist_784', return_X_y=True) #this is handwritten digits data set, 784 pixels (28*28), each pixel is feature
# (784 features to predict 1 / 10 classes )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pca = PCA(n_components=10) #reduce down to 10 principal components

X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

print(X_train)
print(X_train_reduced)

clf = LogisticRegression(max_iter=100)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))


clf = LogisticRegression(max_iter=100)
clf.fit(X_train_reduced, y_train)
print(clf.score(X_test_reduced, y_test))

print(pca.explained_variance_ratio_) #???
print(sum(pca.explained_variance_ratio_))




#METRICS
#CLassification

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit(X_test)

clf = KNeighborsClassifier()
clf.fit(X_train_scaled, y_train)

clf.score(X_test_scaled, y_test) #Accuracy score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = clf.predict(X_test_scaled)
accuracy_score(y_test, y_pred)

precision_score(y_test, y_pred)

recall_score(y_test, y_pred)

f1_score(y_test, y_pred)

#Regression
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

reg = LinearRegression()
reg.fit(X_train_scaled, y_train)

reg.score(X_test_scaled, y_test) #R squared score

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
y_pred = reg.predict(X_test_scaled)

r2_score(y_test, y_pred)

mean_absolute_error(y_test, y_pred)

mean_squared_error(y_test, y_pred)

root_mean_squared_error(y_test, y_pred)


#CROSS VALIDATION
from sklearn.model_selection import cross_val_score
clf = KNeighborsClassifier()

scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='precision') #classifier, scaled data, ground truth, using 5 folds
# do cross validation based on the precision metric??? don't understand
print(scores)

import numpy as np
np.mean(scores)




#HYPERPARAMETER TUNING can be the concept of cross validation
from sklearn.model_selection import GridSearchCV

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

clf = RandomForestClassifier(n_jobs=-1) #n_jobs=-1: this means use all the CPU cores, don't limit to a single core

grid = GridSearchCV(clf, param_grid, cv=3) #classifier, parameter grid, k-fold cross validation

grid.fit(X_train, y_train)

print(grid.best_params_)

best_clf = grid.best_estimator_

best_clf.score(X_test, y_test)


#PIPELINES
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('forest', RandomForestClassifier())
])

pipe.fit(X_train, y_train)

pipe.score(X_test, y_test)