import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
parser = argparse.ArgumentParser('Plot csv data from stdin')
parser.add_argument('--name', default='plot')
parser.add_argument('--x', default='x')
parser.add_argument('--y', default='y')

args = parser.parse_args()


x, y = np.loadtxt(sys.stdin, delimiter=' ', unpack=True)
plt.plot(x, y)

plt.xlabel(f'{args.x}')
plt.ylabel(f'{args.y}')
plt.title(f'{args.name}')
# plt.legend()
plt.savefig(f'{args.name}.png')
