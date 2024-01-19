import sys
import pickle
import numpy as np

def main():
    x = input('Enter an independent value: ')
    try:
        if not x or float(x) < 0:
            raise ValueError('Please enter a positive number')
    except ValueError as e:
        sys.exit(e)

    with open("models.pickle", "rb") as f:
        thetas = pickle.load(f)

    theta0 = float(thetas[0])
    theta1 = float(thetas[1])

    prediction = theta0 + (theta1 * float(x))
    print(prediction)
if __name__ == "__main__":
    main()