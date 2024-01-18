import numpy as np

class LinearRegression:
    def __init__(self, thetas, alpha=0.001, max_iterations=1000):
        self.alpha = alpha
        self.thetas = thetas
        self.max_iterations = max_iterations

    @staticmethod
    def gradient(x, y, thetas):
        m = x.shape[0]

        x_prime = np.column_stack((np.ones_like(x), x))

        # Computing prediction
        prediction = np.dot(x_prime, thetas)

        # Computing difference
        difference = prediction - y

        return np.dot(x_prime.T, difference) / m

    def prediction(self, x):
        x_prime = np.column_stack((np.ones_like(x), x))
        return np.dot(x_prime, self.thetas)

    def fit(self, x, y):
        self.thetas = self.thetas.astype(float).copy()
        for _ in range(self.max_iterations):
            gradient = LinearRegression.gradient(x, y, self.thetas)
            self.thetas -= self.alpha * gradient
        return self.thetas