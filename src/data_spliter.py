import numpy as np

def data_spliter(x, y, proportion=0.8, random_seed=None):
    data = np.column_stack((x, y))

    if random_seed is not None:
        np.random.seed(random_seed)

    np.random.shuffle(data)

    split_index = int(len(data) * (1 - proportion))

    train_data, test_data = data[:split_index], data[split_index:]

    x_train, y_train = train_data[:, 0], train_data[:, 1]
    x_test, y_test = test_data[:, 0], test_data[:, 1]

    return x_train, x_test, y_train, y_test
