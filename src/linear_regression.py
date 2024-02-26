import pickle
import numpy as np
from src.normalization import normalize_min_max

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

            gradient_0 = (1 / m) * np.sum(predictions - prices)
            gradient_1 = (1 / m) * np.sum((predictions - prices) * mileages)

            thetas[0] -= self.learning_rate * gradient_0
            thetas[1] -= self.learning_rate * gradient_1

        self.thetas = thetas
        return error_history

    def save_parameters(self, x, y):
        parameters = {}
        parameters['theta_0'] = self.thetas[0]
        parameters['theta_1'] = self.thetas[1]
        parameters['x_minimum'] = min(x)
        parameters['x_maximum'] = max(x)
        parameters['y_minimum'] = min(y)
        parameters['y_maximum'] = max(y)

        with open("thetas.pickle", "wb") as f:
            pickle.dump(parameters, f)