import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from common.common import init_logging

init_logging(file='lab2/logs/lab2.log.' + str(datetime.now()))

decimals = 3
m = 128
delta = 0.25
data = np.loadtxt('lab2/data/data.txt', delimiter='\t', dtype=int)

autocorellation = np.array([1] + [np.corrcoef(data[:-i], data[i:])[0, 1] for i in range(1, m)])
CC0 = np.argmax(autocorellation < 0) * delta
CC1 = autocorellation[1]
logging.info('CCO=' + np.around(CC0, decimals=decimals).astype(str))
logging.info('CC1=' + np.around(CC1, decimals=decimals).astype(str))

_, axes = plt.subplots(1, 2)
axes[0].set_title('Autocorrelation')
axes[0].plot(np.arange(0, m * delta, delta), autocorellation)
axes[1].set_title('Scattergram')
data_min = data.min()
data_max = data.max()
data_mean = data.mean()
data_limit = data_max * 1.1
ellipse_width = data_max - data_min
ellipse_height = (data_max - data_min) * 1.5
axes[1].add_patch(Ellipse((data_mean, data_mean), ellipse_width, ellipse_height, angle=-45,
                          edgecolor='k', linestyle='--', fc='None', alpha=0.5, lw=0.5))
axes[1].plot([0, data_limit], [0, data_limit], 'k--', alpha=0.5, lw=0.5)
axes[1].scatter(data[:-1], data[1:], s=10)
axes[1].set_xlim(0, data_limit)
axes[1].set_ylim(0, data_limit)
plt.show()
