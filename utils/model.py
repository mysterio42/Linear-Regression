import torch
import torch.nn as nn
from torch.optim import SGD

from utils.data import generate_model_name

WEIGHT_DIR = 'weights'


def load_model(model, path_extension: str):
    return model.load_state_dict(torch.load(path_extension))


def train_model(model, learning_rate, x_train, y_train, save_model: bool):
    optimizer = SGD(params=model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    epochs = 100
    for epoch in range(1, epochs + 1):
        inputs = torch.from_numpy(x_train)
        labels = torch.from_numpy(y_train)

        # clear gradients w.r.t parameters
        optimizer.zero_grad()

        # Forward to get output
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # getting gradients w.r.t parameters
        loss.backward()

        # updating parameters
        optimizer.step()

        print('epoch {} loss {}'.format(epoch, loss.item()))

    if save_model:
        model_name = generate_model_name(5)
        torch.save(model.state_dict(), WEIGHT_DIR + '/' + 'model-' + model_name + '.pkl')
