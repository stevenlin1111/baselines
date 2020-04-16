import argparse
from baselines.common import plot_util as pu

parser = argparse.ArgumentParser()
parser.add_argument('path', help='an integer for the accumulator')
args = parser.parse_args()
exp_prefix = '/home/murtaza/research/baselines/logs/'
results = pu.load_results(exp_prefix + args.path)
import matplotlib.pyplot as plt
import numpy as np
r = results[0]
plt.plot(r.progress.epoch, r.progress['test/success_rate'])
plt.show()
