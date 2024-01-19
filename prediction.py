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

if __name__ == "__main__":
    main()