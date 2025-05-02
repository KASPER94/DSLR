import argparse
from DS.describe import start
from DS.histogram import hist_start
from DS.pair_plot import pair_plot_start
from DS.scatter_plot import scatter_plot_start
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--describe', type=str, help='Program to describe dataset as a parameter')
    parser.add_argument('-v', '--visualization', nargs=2, type=str, help='Program to visualize the data: --v [Histogram, Plot or Scatter] dataset_file')
    args = parser.parse_args()

    if (args.describe):
        start(args.describe)
    if (args.visualization):
        match args.visualization[0]:
            case "Histogram":
                hist_start(args.visualization[1])
            case "Scatter":
                scatter_plot_start(args.visualization[1])
            case "Pair":
                pair_plot_start(args.visualization[1])

if __name__ == "__main__":
    main()