import argparse

from net.LinearRegression import LinearRegressionModel
from utils.data import prepare_data
from utils.model import train_model, load_model
from utils.plot import plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train or load model', type=bool, default=False)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.01)
    parser.add_argument('--state', help='save model state after training', type=bool, default=True)
    args = parser.parse_args()

    x_train, y_train = prepare_data()
    model = LinearRegressionModel(input_dim=1, output_dim=1)
    if args.train:
        train_model(model, learning_rate=args.lr, x_train=x_train, y_train=y_train, save_model=args.state)
        plot(model.train(False), x_train, y_train)
    else:
        model_name = 'model-7pobg.pkl'
        load_model(model, 'weights/{}'.format(model_name))
        plot(model, x_train, y_train)
