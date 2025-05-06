import argparse
from DS.describe import start
from DS.histogram import hist_start
from DS.pair_plot import pair_plot_start
from DS.scatter_plot import scatter_plot_start
from LR.logreg_train import train
from LR.logreg_predict import predict
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--describe', type=str, help='Program to describe dataset as a parameter')
    # parser.add_argument('-v', '--visualization', nargs=2, type=str, help='Program to visualize the data: --v [Histogram, Plot or Scatter] dataset_file')
    parser.add_argument(
        '-v', '--visualization',
        nargs='+',
        type=str,
        help='--v [Histogram | Pair | Scatter] dataset_file [optional: feature_x feature_y]'
    )
    parser.add_argument('-t', '--train', nargs=1, type=str, help='Program to train model: --t dataset_file_train')
    parser.add_argument('-p', '--predict', nargs=1, type=str, help='Program to use model to predict houses: --p dataset_file_test')
    args = parser.parse_args()

    if (args.describe):
        start(args.describe)
    elif (args.visualization):
        match args.visualization[0]:
            case "Histogram":
                hist_start(args.visualization[1])
            case "Scatter":
                if len(args.visualization) >= 4:
                    scatter_plot_start(args.visualization[1], args.visualization[2], args.visualization[3])
                else:
                    scatter_plot_start(args.visualization[1])
            case "Pair":
                    pair_plot_start(args.visualization[1])
    elif (args.train):
        train(args.train[0])
    elif (args.predict):
        predict(args.predict)

if __name__ == "__main__":
    main()