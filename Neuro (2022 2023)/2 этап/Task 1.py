# Импорты всего нужного
import numpy as np
from scipy import signal as sig
from numpy.fft import fft
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Загрузка данных (но определенный диапазон, когда чел. открыл глаза)
s = np.load('test_EEG.npy')[8351:13865]
print(len(s))

# Убираем лишние столбики
n = []
for i in range(len(s)):
    n.append(s[i][0])
s = np.array(n)
print(s)

# преобразование Фурье (возможно тут не нужно)
f = fft(s)[2:2756]

# Ищем частоту
xs = []                             # Частоты для Фурье
t = len(s)/512                      # Время, найденое через частоту диск.
for i in range(2754):
    xs.append(i/t)

# фильтр Баттерворта 4 порядка для верхних и нижних частот (то что нужно)
b, a = sig.butter(4, 2, 'highpass', fs=512)
k1 = sig.filtfilt(b, a, s)
b, a = sig.butter(4, 30, 'lowpass', fs=512)
k2 = sig.filtfilt(b, a, k1)
spec1 = fft(k2)
freqs = fftfreq(len(s), 1/512)
plt.plot(freqs[:len(freqs)//2], abs(spec1[:len(spec1)//2]))
plt.show()

# Огибающая сигнала
def h(k):
    y = []
    for o in range(len(k2)):
        yi = 0
        j = 0
        for e in range(k + 1):
            g = o - (e - k / 2)
            if 0 <= g < len(k2):
                yi += abs(k2[int(g)])
                j += 1
        y.append(yi/j)
    return y
y1 = h(50)             # она выше
y2 = h(500)
plt.plot(k2)
plt.plot(y1)
plt.plot(y2)
plt.show()
print(50, max(y1))     # 50 сэмплов
print(500, max(y2))    # 500 сэмплов

# Кол-во сегментов, подходящие по условию (дальше идет работа с ними)
# n = [0]
# yn = [y1[0]]
# ph = y1[0] > y2[0]
# for i in range(len(y1)):
#     if (y1[i] > y2[i]) != ph:
#         ph = not ph
#         n.append(i)
#         yn.append(y1[i])
# p = 51.2

#
# for i in range(1, len(n)):
#     if n[i] - n[i-1] >= p:
#         if y1[(n[i]+n[i-1])//2]>y2[(n[i]+n[i-1])//2]:
#             pn.append([n[i-1], n[i]])
#             pp.append(n[i-1])
#             pp.append(n[i])
#             ya.append(yn[i-1])
#             ya.append(yn[i])
#             t = len(k2[n[i-1]:n[i]+1])/512
#             fp = fft(k2[n[i-1]:n[i]+1])[int(2*t):int(30*t)]
#             mag.append(list(fp).index(max(fp))/t+2)
# ms = sum(mag)/len(mag)
#
# # Стандартное отклонение по формуле
# d = ((1/len(mag)*sum([(mag[i]-ms)**2 for i in range(len(mag))]))**(0.5))
#
# plt.plot(y1)
# plt.plot(y2)
# plt.plot(pp, ya, "ro")
# plt.show()
# print(len(pn))
# print(d)

start = 0
stop = 0
start_ind = []
stop_ind = []

for i in range(len(y1)-1):                                # ищем число сегментов всех и индексы начала и конца
    if y1[i] <= y2[i] and y1[i+1] > y2[i+1]:
        start += 1
        start_ind.append(i+1)
    elif y1[i] >= y2[i] and y1[i+1] < y2[i+1]:
        stop += 1
        stop_ind.append(i+1)
si = start_ind
sti = stop_ind
for i in range(len(stop_ind)):                           # отсеиваем маленькие
    if stop_ind[i] - start_ind[i] < 51.2:
        start = start - 1
        stop = stop - 1
print(start, stop)
S = []
for i in range(stop-1):
    Spec = abs(fft(k2[si[i]:sti[i]]))
    Spec = Spec[:len(Spec)//2]                          # половинка модуля спектра для каждого сегмента

    Freq = fftfreq(len(k2[si[i]:sti[i]]), 1/512)
    Freq = Freq[:len(Freq)//2]                          # то же для шкалы частот

    S.append(np.round(Freq[Spec == Spec.max()][0]))     # ищем на каких частотах макс значения у Spec

mean = sum(S)/len(S)
a = S-mean

sigma = np.sqrt((1/len(S))*sum(a**2))
print(sigma)
