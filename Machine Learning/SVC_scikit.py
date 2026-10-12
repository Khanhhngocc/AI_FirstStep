# Import Modules
import pandas as pd  # to load and manipulate data and for One-Hot Encoding
import numpy as np  # to calculate the mean and standard deviation
import matplotlib.pyplot as plt  # draw graphs
import matplotlib.colors as colors  # for custom colors
from sklearn.utils import resample  # for downsampling
from sklearn.model_selection import train_test_split  # to split data into training and testing sets
from sklearn.preprocessing import StandardScaler  # to scale data
from sklearn.svm import SVC  # build a Support Vector Classifier
from sklearn.model_selection import GridSearchCV  # for hyperparameter tuning
from sklearn.metrics import ConfusionMatrixDisplay  # to draw a confusion matrix
from sklearn.decomposition import PCA  # for Principal Component Analysis
import ssl
import urllib.request

# 1. IMPORT THE DATA
# Download the file from URL if not already present
try:
    df = pd.read_excel('credit_card_default.xls', header=1)
except FileNotFoundError:
    context = ssl._create_unverified_context()
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
    response = urllib.request.urlopen(url, context=context)
    with open('credit_card_default.xls', 'wb') as f:
        f.write(response.content)
    df = pd.read_excel('credit_card_default.xls', header=1)

print(df.head())

# Rename column and drop ID
df.rename({'default payment next month': 'DEFAULT'}, axis=1, inplace=True)  # Sửa axis='column' thành axis=1
df.drop('ID', axis=1, inplace=True)

# Check unique values
print(df['SEX'].unique())
print(df['EDUCATION'].unique())
print(df['MARRIAGE'].unique())

# Check rows with EDUCATION=0 or MARRIAGE=0
print(len(df.loc[(df['EDUCATION'] == 0) | (df['MARRIAGE'] == 0)]))
print(len(df))

# Remove rows with EDUCATION=0 or MARRIAGE=0
df_no_missing = df.loc[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]  # Sửa logic: dùng & thay vì |

print(df_no_missing['EDUCATION'].unique())
print(df_no_missing['MARRIAGE'].unique())

# Downsample to balance classes
df_no_default = df_no_missing[df_no_missing['DEFAULT'] == 0]
df_default = df_no_missing[df_no_missing['DEFAULT'] == 1]

# Ensure enough samples for downsampling
print(f"No default: {len(df_no_default)}, Default: {len(df_default)}")
if len(df_default) < 1000 or len(df_no_default) < 1000:
    raise ValueError("Not enough samples for downsampling to 1000 per class")

df_no_default_downsampled = resample(df_no_default, replace=False, n_samples=1000, random_state=42)
df_default_downsampled = resample(df_default, replace=False, n_samples=1000, random_state=42)

df_downsample = pd.concat([df_no_default_downsampled, df_default_downsampled])

# Prepare features and target
X = df_downsample.drop('DEFAULT', axis=1).copy()
y = df_downsample['DEFAULT'].copy()

# One-hot encoding for categorical columns
X_encoded = pd.get_dummies(X, columns=['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'])

# Check unique values of PAY_* columns
for col in ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
    print(f"{col} unique values: {X[col].unique()}")

# Split data and scale
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train initial SVC
clf_svm = SVC(random_state=42)
clf_svm.fit(X_train_scaled, y_train)

# Draw confusion matrix
ConfusionMatrixDisplay.from_estimator(
    clf_svm,
    X_test_scaled,
    y_test,
    values_format='d',
    display_labels=["Did not default", "Defaulted"]
)
plt.show()

# Tune hyperparameters with GridSearchCV
param_grid = [
    {'C': [0.5, 1, 10, 100],
     'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
     'kernel': ['rbf']},
]

optimal_params = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=0
)
optimal_params.fit(X_train_scaled, y_train)
print(optimal_params.best_params_)

# Train SVC with optimal parameters
clf_svm_optimal = SVC(random_state=42, **optimal_params.best_params_)
clf_svm_optimal.fit(X_train_scaled, y_train)

# Draw confusion matrix for optimized SVC
ConfusionMatrixDisplay.from_estimator(
    clf_svm_optimal,
    X_test_scaled,
    y_test,
    values_format='d',
    display_labels=["Did not default", "Defaulted"]
)
plt.show()

# Apply PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)

# Draw Scree Plot
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = [str(x) for x in range(1, len(per_var) + 1)]

plt.bar(x=range(1, len(per_var) + 1), height=per_var)
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Components')
plt.title('Scree Plot')
plt.show()

# Train SVC on first two principal components
train_pc1_coords = X_train_pca[:, 0]
train_pc2_coords = X_train_pca[:, 1]
pca_train_scaled = scaler.fit_transform(np.column_stack((train_pc1_coords, train_pc2_coords)))  # Sửa: dùng scaler thay vì scale

param_grid = [
    {'C': [1, 10, 100, 1000],
     'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
     'kernel': ['rbf']},
]
optimal_params = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=0
)
optimal_params.fit(pca_train_scaled, y_train)
print(optimal_params.best_params_)

clf_svm = SVC(random_state=42, **optimal_params.best_params_)
clf_svm.fit(pca_train_scaled, y_train)

# Visualize decision boundary on PCA-transformed test data
X_test_pca = pca.transform(X_test_scaled)
test_pc1_coords = X_test_pca[:, 0]
test_pc2_coords = X_test_pca[:, 1]

x_min = test_pc1_coords.min() - 1
x_max = test_pc1_coords.max() + 1
y_min = test_pc2_coords.min() - 1
y_max = test_pc2_coords.max() + 2

xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1),
                     np.arange(start=y_min, stop=y_max, step=0.1))

Z = clf_svm.predict(np.column_stack((xx.ravel(), yy.ravel())))
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 10))
ax.contour(xx, yy, Z, alpha=0.1)

cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])

scatter = ax.scatter(test_pc1_coords, test_pc2_coords, c=y_test,
                     cmap=cmap,
                     s=100,
                     edgecolors='k',
                     alpha=0.7)

legend = ax.legend(scatter.legend_elements()[0],
                   scatter.legend_elements()[1],
                   loc="upper right")
legend.get_texts()[0].set_text("No Default")
legend.get_texts()[1].set_text("Yes Default")

ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_title('Decision surface using the PCA transformed/projected features')
plt.show()