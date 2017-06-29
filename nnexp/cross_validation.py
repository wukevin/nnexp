"""
Runs k-fold cross validation on the given
"""
import argparse
import os
import shlex
import subprocess

def run_cnn(k, size=40):
    """
    Runs the convolution neural network on the given parameters
    """
    binary = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tcga_nn.py")
    assert os.path.isfile(binary)
    command = "python3 {0}".format(binary)
    command += " --kth {0} --size {1}".format(k, size)
    print(command)
    command_tokenized = shlex.split(command)
    subprocess.call(command_tokenized)

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", default=568, type=int, help="Total number of datapoints that there are")
    parser.add_argument("--chunk", default=40, type=int, help="Size of each chunk to process")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    for i in range(int(round(args.total / args.chunk))):
        run_cnn(i, args.chunk)

if __name__ == "__main__":
    main()
