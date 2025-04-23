import argparse
from describe import start

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--describe', type=str, help='Prog to describe dataset as a paramete')
    parser.add_argument('-v', '--visualization', type=str, help='Prog to describe dataset as a paramete')
    args = parser.parse_args()

    if (args.describe):
        start(args.describe)
    if (args.visualization):
        match args.visualization:
            case "Histogram":
                print("HELL")
                pass
            case "Scatter":
                pass
            case "Pair":
                pass

if __name__ == "__main__":
    main()