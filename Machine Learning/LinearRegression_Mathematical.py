import pandas as pd
import matplotlib as plt

data = pd.read_csv('')

def loss_function(m, b, points):
    total_error = 0
    for i in range (len(points)):
        total_error += (points[i].score - (m * points[i].studytime + b))
    return total_error / float(len(points))

def gradien_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range (n):
        x = points[i].studytime
        y = points[i].score

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient


m = 0
b= 0
L = 0.0001
epochs = 1000

for i in range(epochs):
    if( i % 50 == 0):
        print(f"Epoch:  {i}")
    m, b = gradien_descent(m, b, data, L)

print(m, b)

plt.scatter(data.studytime, data.score, color="black")
plt.plot(list(range(20, 80)), [m * x + b for x in range(20, 80)], color="red")
plt.show()

