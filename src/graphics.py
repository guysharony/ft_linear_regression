import matplotlib.pyplot as plt

def plot_dataset(x, y, y_prediction):
    plt.scatter(x, y, color="blue", label="Data points", zorder=3)
    plt.plot(x, y_prediction, color="red", label="Prediction", zorder=3)
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, zorder=0)
    plt.show()