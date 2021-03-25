import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str)
parser.add_argument('--shape', '-s', action='store_true')
parser.add_argument('--value', '-v', action='store_true')
args = parser.parse_args()

a = np.load(args.file)

np.set_printoptions(3)
if args.shape:
    print(f'shape = {a.shape}')

if args.value:
    print(f'value = {a}')



