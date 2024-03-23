import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fftpack import fft

data_closed = np.loadtxt('ЗакрытыеГлаза.dat')
data_closed_mental = np.loadtxt('ЗакрытыеГлазасУмственнойнагрузкой.dat')

fs = 250
n = len(data_closed)

def calc_power(data, fs):
    power_bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (14, 40)}
    freqs, psd = welch(data, fs, nperseg=1024)
    power_dict = {}
    for band, freq_range in power_bands.items():
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        power_dict[band] = np.trapz(psd[freq_mask], freqs[freq_mask])
    return power_dict

power_closed = np.array([calc_power(data_closed[:,i], fs) for i in range(data_closed.shape[1])])
power_closed_mental = np.array([calc_power(data_closed_mental[:,i], fs) for i in range(data_closed_mental.shape[1])])

channel_labels = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5', 'Channel 6', 'Channel 7', 'Channel 8']
print("Мощность сигналов для состояния с закрытыми глазами:")
for i, label in enumerate(channel_labels):
    print(f"{label}: Delta: {power_closed[i]['Delta']:.2f}, Theta: {power_closed[i]['Theta']:.2f}, Alpha: {power_closed[i]['Alpha']:.2f}, Beta: {power_closed[i]['Beta']:.2f}")

print("\nМощность сигналов для состояния с закрытыми глазами при умственной нагрузке:")
for i, label in enumerate(channel_labels):
    print(f"{label}: Delta: {power_closed_mental[i]['Delta']:.2f}, Theta: {power_closed_mental[i]['Theta']:.2f}, Alpha: {power_closed_mental[i]['Alpha']:.2f}, Beta: {power_closed_mental[i]['Beta']:.2f}")

