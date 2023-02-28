#s11163068_Tomasi Junior
#CS415 Lab 1 Week 3

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, lr=0.1, n_iter=100):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)
        for _ in range(self.n_iter):
            for i in range(X.shape[0]):
                y_hat = self.predict(X[i])
                update = self.lr * (y[i] - y_hat)
                self.weights[0] += update
                self.weights[1:] += update * X[i]

    def predict(self, X):
        z = np.dot(X, self.weights[1:]) + self.weights[0]
        return np.where(z > 0, 1, -1)


def generate_data(n_samples=60):
    X = np.random.rand(n_samples, 2) * 2 - 1
    y = np.ones(n_samples)
    y[np.sum(X, axis=1) < 0] = -1
    return X, y


X, y = generate_data()

p = Perceptron(lr=0.1, n_iter=100)
p.fit(X, y)

# plot the data and decision boundary
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
Z = np.sign(p.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
ax.contour(xx, yy, Z, levels=[0], colors='k')
plt.show()
