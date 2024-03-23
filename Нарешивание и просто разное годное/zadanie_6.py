import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as ppc
from scipy import signal as sig

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    return b, a

# функция полосно-пропускающего фильтра
# data - исходные данные
# lowcut - "нижняя граница": частота, ниже которой частоты в наши данные пропускаться не будут 
# highcut  - "верхняя граница": частота, выше которой частоты в наши данные пропускаться не будут  
# fs - частота оцифровки данных
# order - "порядок фильтра" - влияет на степень фильтрации, а также величину артефактов фильтрации
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y

raw_data = np.loadtxt( 'data_prepare.dat', delimiter='\t', dtype=np.int)
raw_data_tr = np.transpose(raw_data)
data1 = raw_data_tr[1]
data2 = raw_data_tr[2]

# в результате шкалирования данныех - среднее по каждому каналу данных сдвигается к нулю, а амплитуды сжимаются так, чтобы среднеквадратичное отклонение стало равно единице
data1_scaled = ppc.scale(data1)
data2_scaled = ppc.scale(data2)

# отобразим результаты шкалирования на графике
fig, axs = plt.subplots(2)
axs[0].plot(data1, color='tab:green', label='исходные данные 1')
axs[0].plot(data2, color='tab:orange', label='исходные данные 2')
axs[1].plot(data1_scaled, color='tab:green', label='шкалированные данные 1')
axs[1].plot(data2_scaled, color='tab:orange', label='шкалированные данные 2')
for ax in axs:
    ax.legend(loc='lower left', shadow=True, fontsize='x-large')
plt.show()
# Видно, что после шкалирования совершенно разные в численном отношении, но похожие по общей форме данные становятся действительно похожими

# т.к. мы знаем длительность пика Р300 по времени, мы можем приблизительно оценить, из сигналов с какой частотой он состоит
# Р300 достаточно длительный по времени и пологий, значит очевидно высокие частоты не играют важной роли в формировании искомого сигнала, поэтому все частоты выше 20Гц мы отфильтруем 
# длина Р300 по времени составляет менее секунды, значит очевидно высокие частоты ниже 0.5Гц также не играют важной роли в формировании искомого сигнала, но могут вносить смещение распознаваемого участка по амплитуде, поэтому частоты ниже 1.0Гц также отфильтруем  
data1_filtered = butter_bandpass_filter(data1, 1.0, 20, 240)

# отобразим результаты фильтрации на графике
fig, axs = plt.subplots()
axs.plot(data1, color='tab:green', label='исходные данные')
axs.plot(data1_filtered, color='tab:blue', label='фильтр от 1.0 до 20Гц')
axs.legend(loc='lower left', shadow=True, fontsize='x-large')
plt.show()
# видно, что среднее значение данных сместилось к нулю, а также исчезли или сгладились острые "выбросы" на данных. Также видно, что за счет краевых эффектов фильтрации график в начальной части кривой несколько потерял свой вид. При использовании фильтров необходимо контролировать по тестовым данным наличие краевых и других воможных артефактов фильтрации.  

# т.к. пик Р300 достаточно длительный по времени, а оцифрован с частотой 240Гц, большое количество точек могут приводить к дополнительным ошибкам в распознавании .  Уменьшим количество точек.
# этот процесс называется децимацией, и может в некотором роде заменить фильтрацию высоких частот, кроме того существенно уменьшает объемы данных, а значит увеличивает быстродействие 
data_decimated = sig.decimate(data1, 5, ftype='fir', axis=0)

# отобразим результаты децимации на графике
fig, axs = plt.subplots()
axs.plot(data1, color='tab:green', label='исходные данные')
axs.scatter(range(0, len(data1)), data1, color='tab:green')
axs.plot( range(0, len(data1), 5), data_decimated, color='tab:orange', label='после децимации')
axs.scatter( range(0, len(data1), 5), data_decimated, color='tab:orange')
axs.legend(loc='lower left', shadow=True, fontsize='x-large')
plt.show()
# видно, что общая форма графика практически не изменилась, однако исчезли острые "выпадающие" точки, и общее количество точек уменьшилось

