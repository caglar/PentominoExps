import argparse
import matplotlib as mpl
mpl.use('Agg')
from decimal import Decimal
from matplotlib import pyplot as plt
import numpy as np
import cPickle as  pkl

parser = argparse.ArgumentParser(description="A script to plot the learning curves.")
parser.add_argument("--mode", type=str, default="Train")
args = parser.parse_args()

assert args.mode in ["Train", "Valid"], "mode should be either training or validation."

mode = args.mode

stats_nohints = pkl.load(open("pento_nohints_rmsprop_stats.pkl"))
stats_hints = pkl.load(open("pento_hints_rmsprop_stats.pkl"))
hints_full_costs = stats_hints["%s_full_costs" % mode]
nohints_full_costs = stats_nohints["%s_full_costs" % mode]

fig, ax = plt.subplots()

plt.plot(nohints_full_costs)
plt.plot(hints_full_costs)
fig.canvas.draw()

labels = ["%.1E" % Decimal(i*50*400*200) for i, t in enumerate(ax.get_xticklabels())]
plt.ticklabel_format(axis='x', style='sci', scilimits=(1, 2))
plt.gca().set_xticklabels(labels)


plt.xlabel("# Examples seen")
plt.ylabel("%s cost" % mode)
plt.legend(["No hints %s cost(nll)" % mode, "Hints %s cost(nll)" % mode], prop={'size': 6})
plt.show()
plt.savefig("%s_costs_learning_curves.pdf" % mode)


