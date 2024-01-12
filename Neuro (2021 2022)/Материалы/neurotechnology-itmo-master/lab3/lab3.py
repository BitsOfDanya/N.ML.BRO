import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from common.common import init_logging

init_logging(file='lab3/logs/lab3.log.' + str(datetime.now()))

decimals = 3
m = 128
delta = 0.25
data = np.loadtxt('lab3/data/data.txt', delimiter='\t', dtype=int)

N = data.shape[0]
Y = data
X = np.array([np.sum(data[:i + 1]) for i in range(N)])
f = X / X[N - 1]
F = np.array([1 / N * np.sum(Y * np.exp(-(2 * np.pi * 1j / N * k * (np.arange(N) + 1)))) for k in (np.arange(N) + 1)])
P = np.square(np.real(F)) + np.square(np.imag(F))

P_half = P[:150]
f_half = f[:150]

plt.title('Spectral power density')
plt.plot(f_half, P_half)
plt.show()

HF = np.sum(P[np.logical_and(f > 0.15, f <= 0.4)])
LF = np.sum(P[np.logical_and(f > 0.04, f <= 0.15)])
VLF = np.sum(P[np.logical_and(f > 0.015, f <= 0.04)])
TP = HF + LF + VLF

logging.info('HF=' + np.around(HF, decimals=decimals).astype(str))
logging.info('LF=' + np.around(LF, decimals=decimals).astype(str))
logging.info('VLF=' + np.around(VLF, decimals=decimals).astype(str))
logging.info('TP=' + np.around(TP, decimals=decimals).astype(str))

HF_percentage = HF / TP * 100
LF_percentage = LF / TP * 100
VLF_percentage = VLF / TP * 100

logging.info('HF%=' + HF_percentage.astype(int).astype(str) + '%')
logging.info('LF%=' + LF_percentage.astype(int).astype(str) + '%')
logging.info('VLF%=' + VLF_percentage.astype(int).astype(str) + '%')

IC = (VLF + LF) / HF
IVV = LF / HF
ISCA = LF / VLF

logging.info('IC=' + np.around(IC, decimals=decimals).astype(str))
logging.info('IVV=' + np.around(IVV, decimals=decimals).astype(str))
logging.info('ISCA=' + np.around(ISCA, decimals=decimals).astype(str))
