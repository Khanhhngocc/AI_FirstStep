import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


X, y = load_breast_cancer(return_X_y=True)
print("Shape X:", X.shape)  # (569, 30)
print("Label 0 là:", load_breast_cancer().target_names[0])  # malignant
print("Label 1 là:", load_breast_cancer().target_names[1])  # benign


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)  # thêm max_iter nếu dữ liệu lớn
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Độ chính xác (accuracy):", acc)
print("Classification Report:\n", classification_report(y_test, y_pred))
