import pandas as pd
from src.data_spliter import data_spliter
from src.linear_regression import LinearRegression

def main():
    try:
        dataset = pd.read_csv("./dataset/data.csv")

        x = dataset['km'].values
        y = dataset['price'].values

        x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

        print(x_train, x_test, y_train, y_test)
    except Exception as err:
        print(f'Error: {err}')

if __name__ == "__main__":
    main()