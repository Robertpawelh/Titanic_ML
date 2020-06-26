import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, w=None, lbd=0.003, eta=0.4, epochs=2000, theta=0.55):
        self.w = w
        self.lbd = lbd
        self.eta = eta
        self.epochs = epochs
        self.theta = theta

    def add_column_with_ones(self, x):
        N, M = x.shape
        x_train = np.ones((N, M + 1))
        x_train[:, 1:] = x
        return x_train

    def regularized_logistic_cost_function(self, w, x_train, y_train):
        N, M = x_train.shape
        sigma = sigmoid(x_train @ w)
        costFunction = -1 / N * np.sum(y_train * np.log(sigma) + (1 - y_train) * np.log(1 - sigma)) + (self.lbd / 2) * (
                    np.linalg.norm(w[1:]) ** 2)
        w_0 = w.copy()
        w_0[0] = 0
        grad = 1 / N * x_train.T @ (sigma - y_train) + self.lbd * w_0
        return costFunction, grad

    def gradient_descent(self, x_train, y_train, w0):
        w = w0
        for k in range(self.epochs):
            _, grad = self.regularized_logistic_cost_function(w, x_train, y_train)
            w = w - self.eta * grad
        return w

    def fit(self, x_train, y_train):
        x_train = self.add_column_with_ones(x_train)
        w0 = np.zeros(x_train.shape[1])
        self.w = self.gradient_descent(x_train, y_train, w0)

    def predict(self, x):
        x = self.add_column_with_ones(x)
        N = x.shape[0]
        sigm = sigmoid(x @ self.w)
        return np.array([int((sigm[i] >= self.theta)) for i in range(N)])
