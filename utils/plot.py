import matplotlib.pyplot as plt
import torch


def plot(model, x_train, y_train):
    plt.clf()

    predicted = model(torch.from_numpy(x_train)).detach().numpy()

    plt.plot(x_train, y_train, 'go', label='True Data', alpha=0.5)
    plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
    plt.legend(loc='best')
    plt.show()
