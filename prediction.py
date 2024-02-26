import os
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

    parameter_keys = ['theta_0', 'theta_1', 'x_minimum', 'x_maximum', 'y_minimum', 'y_maximum']

    if not os.path.isfile('thetas.pickle'):
        parameters = {key: 0 for key in parameter_keys}
    else:
        try:
            with open("thetas.pickle", "rb") as f:
                parameters = pickle.load(f)
                parameters = {key: float(parameters[key]) for key in parameter_keys}
        except Exception as error:
            print(f"Error: {error}")
            sys.exit(1)

        # Normalizing input
        x = normalize_min_max(x, parameters['x_minimum'], parameters['x_maximum'])

    # Predicting price
    normalized_prediction = parameters['theta_0'] + (parameters['theta_1'] * x)

    # De-normalizing price
    denormalized_price = denormalize_min_max(normalized_prediction, parameters['y_minimum'], parameters['y_maximum'])

    # Display predicted price
    print('The price will be:', denormalized_price)

if __name__ == "__main__":
    main()