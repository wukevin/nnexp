"""
Runs k-fold cross validation on the given
"""
import argparse
import os
import shlex
import subprocess
import re
import glob

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

def parse_cnn_logs(filename):
    """Given a cnn log, return the number of true positives, and total test cases"""
    with open(filename, 'r') as handle:
        lines = [l.rstrip() for l in handle]
    summary_line = lines[-1]
    fraction = re.findall(r"[0-9]. \/ [0-9].$", summary_line).pop()
    tp, total = fraction.split("/")
    tp, total = int(tp), int(total)
    return tp, total

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", default=568, type=int, help="Total number of datapoints that there are")
    parser.add_argument("--chunk", default=40, type=int, help="Size of each chunk to process")
    parser.add_argument("--output", default="cross_validation.log", type=str, help="output file to write summary to")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    for i in range(int(round(args.total / args.chunk))):
        run_cnn(i, args.chunk)
    
    # Parse the logs to summarize the CNN
    tp, total = 0, 0
    for logfile in glob.glob("cnn.*.log"):
        x, y = parse_cnn_logs(logfile)
        tp += x
        total += y

    with open(args.output, 'w') as handle:  # Write final cross-validation results
        handle.write("%i / %i\n" % (tp, total))

if __name__ == "__main__":
    main()
