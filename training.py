import sys
import pickle
import pandas as pd

from src.graphics import plot_training
from src.normalization import normalize_min_max
from src.normalization import denormalize_min_max
from src.LinearRegression import LinearRegression


def main():
    try:
        if len(sys.argv) != 2:
            print('Usage: python training.py <csv_file_path>')
            sys.exit(1)

        dataset = pd.read_csv(sys.argv[1])

        x = dataset["km"].values
        y = dataset["price"].values

        # Mileage minimum and maximum
        x_minimum, x_maximum = min(x), max(x)

        # Prices minimum and maximum
        y_minimum, y_maximum = min(y), max(y)

        # Normalizing mileages
        x_normalized = normalize_min_max(x, x_minimum, x_maximum)
        y_normalized = normalize_min_max(y, y_minimum, y_maximum)

        # Initializing model
        model = LinearRegression(
            learning_rate=0.1,
            max_iterations=1_000,
        )

        # Gradient descent
        errors = model.gradient_descent(x_normalized, y_normalized)
        print(f"MSE Evolution [{errors[0]}] -> [{errors[-1]}]")

        # Saving parameters
        model.save_parameters(x, y)

        # Predictions
        y_predictions_normalized = model.prediction(x_normalized)

        # De-normalizing predictions
        y_predictions_denormalized = denormalize_min_max(y_predictions_normalized, y_minimum, y_maximum)

        # Plotting data
        plot_training(x, y, y_predictions_denormalized, errors)
    except Exception as err:
        print(f'Error: {err}')

if __name__ == "__main__":
    main()