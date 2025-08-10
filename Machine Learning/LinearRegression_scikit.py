import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

time_studied = np.array([20, 50, 65])
scores = ([56, 83, 47])

time_train, time_test, score_train, score_test = train_test_split(time_studied, scores, test_size=0.1)

model = LinearRegression()
model.fit(time_train, score_train)

print(model.score(time_train, score_train))

plt.scatter(time_train, score_train)
plt.plot(np.linspace(0, 70, 100).reshape(-1, 1), model.predict(np.linspace(0, 70, 100).reshape(-1, 1)), 'r')
plt.show()