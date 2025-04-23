import argparse
from describe import start

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--describe', type=str, help='Prog to describe dataset as a paramete')
    args = parser.parse_args()

    if (args.describe):
        start(args.describe)

if __name__ == "__main__":
    main()