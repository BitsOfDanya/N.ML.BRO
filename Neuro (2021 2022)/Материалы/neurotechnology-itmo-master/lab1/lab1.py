import logging
from datetime import datetime

import numpy as np

from common.common import init_logging

init_logging(file='lab1/logs/lab1.log.' + str(datetime.now()))

step = 50
decimals = 3
data = np.loadtxt('lab1/data/data.txt', delimiter='\t')

data_min = data.min()
data_max = data.max()
lower_bound = (data_min / 100).astype(int) * 100
upper_bound = (data_max / 100).astype(int) * 100 + 100

intervals = np.append(np.arange(lower_bound, upper_bound, step), upper_bound)
occurrences = np.histogram(data, bins=intervals + 1)[0]
frequencies = occurrences / data.size

logging.info('#1')
logging.info('intervals: ' + ','.join(intervals.astype(str)))
logging.info('frequencies: ' + ','.join(np.around(frequencies, decimals=decimals).astype(str)))

M = data.sum() / data.size
SDNN = np.sqrt(np.power(data - M, 2).sum() / (data.size - 1))
CV = SDNN / M * 100
RMSSD = np.sqrt(np.power(data[:-1] - data[1:], 2).sum() / (data.size - 1))

logging.info('#2')
logging.info('M = ' + np.around(M, decimals=decimals).astype(str))
logging.info('SDNN = ' + np.around(SDNN, decimals=decimals).astype(str))
logging.info('CV = ' + np.around(CV, decimals=decimals).astype(str))
logging.info('RMSSD = ' + np.around(RMSSD, decimals=decimals).astype(str))

most_interval_index = np.argmax(occurrences)
Mo = (intervals[most_interval_index] + step / 2).astype(int)
AMo = (frequencies[most_interval_index] * 100).astype(int)
MxDMn = (data_max - data_min).astype(int)

logging.info('#3')
logging.info('Mo = ' + Mo.astype(str))
logging.info('AMo = ' + AMo.astype(str))
logging.info('MxDMn = ' + MxDMn.astype(str))

SI = AMo / (2 * Mo / 1000 * MxDMn / 1000)
logging.info('#4')
logging.info('SI = ' + np.around(SI, decimals=decimals).astype(str))
