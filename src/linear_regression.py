import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.001, max_iterations=1000):
        self.thetas = np.zeros(2)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def prediction(self, mileage, thetas=None):
        t = self.thetas if thetas is None else thetas

        return t[0] + (t[1] * mileage)

    def mean_squared_error(self, target, prediction):
        m = len(target)
        return np.sum((target - prediction) ** 2) / m

    def gradient_descent(self, mileages, prices):
        m = len(mileages)
        error_history = []

        thetas = self.thetas.astype(float).copy()
        for _ in range(self.max_iterations):
            predictions = self.prediction(mileages, thetas)

            # Saving error
            mse = self.mean_squared_error(prices, predictions)
            error_history.append(mse)

            gradient_0 = np.sum(predictions - prices) / m
            gradient_1 = np.sum((predictions - prices) * mileages) / m

            thetas[0] -= self.learning_rate * gradient_0
            thetas[1] -= self.learning_rate * gradient_1

        self.thetas = thetas
        return error_history