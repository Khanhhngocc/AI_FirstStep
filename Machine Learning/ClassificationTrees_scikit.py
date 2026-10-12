# Import Modules
import pandas as pd  # to load and manipulate data and for One-Hot Encoding
import numpy as np  # to calculate the mean and standard deviation
import matplotlib.pyplot as plt  # draw graphs
from sklearn.tree import DecisionTreeClassifier  # build a classification tree
from sklearn.tree import plot_tree  # to draw a classification tree
from sklearn.model_selection import train_test_split  # to split data into training and testing sets
from sklearn.model_selection import cross_val_score  # for cross validation
from sklearn.metrics import ConfusionMatrixDisplay  # to draw a confusion matrix

# 1. IMPORT THE DATA
# since we have already downloaded the data to a file, we can load the file...
df = pd.read_csv('processed.cleveland.data.csv')

# print the first 5 rows
df.head()

# change the column numbers to column names (fixed to include 14 columns)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'hd']

# print out unique values in the column called 'ca'
df['ca'].unique()

# print out unique values in the column called 'thal'
df['thal'].unique()

# check rows with missing values ('?')
len(df.loc[(df['ca'] == '?') | (df['thal'] == '?')])

# display rows with missing values
df.loc[(df['ca'] == '?') | (df['thal'] == '?')]

# total number of rows
len(df)

# create new dataframe without missing values
df_no_missing = df.loc[(df['ca'] != '?') & (df['thal'] != '?')]

# make a new copy of the columns used to make predictions
X = df_no_missing.drop('hd', axis=1).copy()  # alternatively: X = df_no_missing.iloc[:,:-1]
X.head()

# make a new copy of the column of data we want to predict
y = df_no_missing['hd'].copy()
y.head()

# one-hot encoding for categorical columns
X_encoded = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'thal'])
X_encoded.head()

# convert y to binary (0 or 1)
y_not_zero_index = y > 0  # get the index for each non-zero value in y
y[y_not_zero_index] = 1  # set each non-zero value in y to 1
y.unique()  # verify that y only contains 0 and 1

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)

# create a decision tree and fit it to the training data
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(X_train, y_train)

# draw a confusion matrix
disp = ConfusionMatrixDisplay.from_estimator(
    clf_dt,
    X_test,
    y_test,
    display_labels=["Does not have HD", "Has HD"]
)
disp.plot()

# determine values for alpha for pruning
path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas  # extract different values for alpha
ccp_alphas = ccp_alphas[:-1]  # exclude the maximum value for alpha

clf_dts = []  # create an array that we will put decision trees into
# create one decision tree per value for alpha and store it in the array
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf_dt.fit(X_train, y_train)
    clf_dts.append(clf_dt)

# calculate train and test scores for each alpha
train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

# plot accuracy vs alpha
fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title('Accuracy vs alpha for training and testing sets')
ax.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle='steps-post')
ax.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle='steps-post')
ax.legend()

# create an array to store the results of each fold during cross validation
alpha_loop_values = []

# for each candidate value for alpha, run 5-fold cross validation
# store the mean and standard deviation of the scores (the accuracy)
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

# draw a graph of the means and standard deviations of the scores for each alpha
alpha_results = pd.DataFrame(alpha_loop_values, columns=['alpha', 'mean_accuracy', 'std'])
alpha_results.plot(x='alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--')

# select ideal alpha
ideal_ccp_alpha = float(alpha_results[(alpha_results['alpha'] > 0.014) & (alpha_results['alpha'] < 0.015)]['alpha'])

# build and train a new decision tree with the optimal alpha
clf_dt_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=ideal_ccp_alpha)
clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

# draw confusion matrix for pruned tree
disp_pruned = ConfusionMatrixDisplay.from_estimator(
    clf_dt_pruned,
    X_test,
    y_test,
    display_labels=["Does not have HD", "Has HD"]
)
disp_pruned.plot()

# visualize the pruned tree
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["No HD", "Yes HD"],
          feature_names=X_encoded.columns)

plt.show()