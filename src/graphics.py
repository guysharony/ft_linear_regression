import matplotlib.pyplot as plt

def plot_dataset(ax, x, y, y_prediction):
    ax.scatter(x, y, color="blue", label="Data points", zorder=3)
    ax.plot(x, y_prediction, color="red", label="Prediction", zorder=3)
    ax.set_xlabel("Mileage (km)")
    ax.set_ylabel("Price")
    ax.set_title("Linear regression")
    ax.legend()
    ax.grid(True, zorder=0)


def plot_error_history(ax, errors):
    x = range(1, len(errors) + 1)

    ax.plot(x, errors, color="red", label="MSE Evolution", zorder=3)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("MSE Evolution on training")
    ax.legend()
    ax.grid(True, zorder=0)


def plot_training(x, y, predictions, errors):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    plot_dataset(axes[0], x, y, predictions)
    plot_error_history(axes[1], errors)

    fig.suptitle("Linear regression training status")
    plt.tight_layout()
    plt.show()