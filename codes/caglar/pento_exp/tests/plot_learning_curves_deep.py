import matplotlib as mpl
mpl.use('Agg')
from decimal import Decimal
from matplotlib import pyplot as plt
import numpy as np
import cPickle as  pkl

stats_nohints = pkl.load(open("pento_mainloop_nohints2_stats.pkl"))

nohints_full_traincosts = stats_nohints['Train_full_costs']
nohints_full_validcosts = stats_nohints['Valid_full_costs']
fig, ax = plt.subplots()

plt.plot(nohints_full_traincosts)
plt.plot(nohints_full_validcosts)
fig.canvas.draw()

labels = ["%.1E" % Decimal(i*50*400*200) for i, t in enumerate(ax.get_xticklabels())]
plt.ticklabel_format(axis='x', style='sci', scilimits=(1, 2))
plt.gca().set_xticklabels(labels)


plt.xlabel("# Examples seen")
plt.ylabel("Training cost")
plt.legend(["Training cost(nll)", "Valid cost(nll)"], prop={'size': 6})
plt.show()
plt.savefig("training_costs_nohints_deepmind.png")

import ipdb; ipdb.set_trace()
