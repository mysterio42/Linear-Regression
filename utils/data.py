import random
import string

import numpy as np


def prepare_data():
    x_values = [i for i in range(11)]
    x_train = np.array(x_values, dtype=np.float32)  # (11,)
    x_train = x_train.reshape(-1, 1)  # (11,1)
    y_values = [2 * i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)
    return x_train, y_train


def generate_model_name(size=5):
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))
