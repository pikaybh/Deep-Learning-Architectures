from models import _01_nn as mynn
import argparse


parser = argparse.ArgumentParser(description='1: nn\n2: cnn\n3: rnn ...')
parser.add_argument('model', type=int, help='', required=True)
args = parser.parse_args()


def main() -> None:
    if args.model == 1:
        mynn

# Main
if __name__ == '__main__':
    main()
