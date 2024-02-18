import sys
import pickle
from src.normalization import normalize_min_max
from src.normalization import denormalize_min_max

def main():
    x = input('Please enter a mileage: ')
    try:
        if not x or not x.isdigit():
            raise ValueError('Mileage should be a number.')

        x = float(x)

        if not x or x < 0:
            raise ValueError('Mileage should be positive.')
    except ValueError as error:
        print(f'error: {error}')
        sys.exit(1)

    try:
        with open("models.pickle", "rb") as f:
            parameters = pickle.load(f)
    except FileNotFoundError:
        print("Error: Paramaters file not found.")
        sys.exit(1)
    except pickle.PickleError as error:
        print(f"Error: Failed to load parameters file{error}")
        sys.exit(1)
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)

    # Normalizing input
    x_minimum = float(parameters['x_minimum'])
    x_maximum = float(parameters['x_maximum'])
    normalize_input = normalize_min_max(x, x_minimum, x_maximum)

    # Predicting price
    theta_0 = float(parameters['theta_0'])
    theta_1 = float(parameters['theta_1'])
    normalized_prediction = theta_0 + (theta_1 * normalize_input)

    # De-normalizing price
    y_minimum = float(parameters['y_minimum'])
    y_maximum = float(parameters['y_maximum'])
    denormalized_price = denormalize_min_max(normalized_prediction, y_minimum, y_maximum)

    # Display predicted price
    print('The price will be: ', denormalized_price)
if __name__ == "__main__":
    main()