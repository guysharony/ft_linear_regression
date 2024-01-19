import pickle
import numpy as np
import pandas as pd
from src.graphics import plot_dataset
from src.normalization import normalize_min_max
from src.linear_regression import LinearRegression


def main():
    dataset = pd.read_csv("./dataset/data.csv")

    x = dataset["km"].values
    y = dataset["price"].values

    # Normalizing mileages
    x_normalized, _, _ = normalize_min_max(x)
    y_normalized, y_minimum, y_maximum = normalize_min_max(y)

    # Initializing model
    model = LinearRegression(
        learning_rate=0.1,
        max_iterations=1_000
    )

    # Gradient descent
    errors = model.gradient_descent(x_normalized, y_normalized)
    print(f"MSE Evolution [{errors[0]}] -> [{errors[-1]}]")

    # Predictions
    y_predictions_normalized = model.prediction(x_normalized)

    # De-normalising predictions for plot
    y_predictions_denormalized = y_predictions_normalized * (y_maximum - y_minimum) + y_minimum

    # Plotting data
    plot_dataset(x, y, y_predictions_denormalized)

if __name__ == "__main__":
    main()