import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

traindf = pd.read_csv('ECG_train.csv')
testdf = pd.read_csv('ECG_test.csv')
hz = 1000 # частота оцифровки сигнала

# Функции для работы с ЭКГ
def getPeaks(ecg, thold=0.3):
    """Вычисление индексов точек, соответствующих вершинам R-зубцов
    thold - пороговое значение сигнала выше которого будут искаться
    пики"""
    i = 0
    peaks = []
    while i < len(ecg) - 1:
        if ecg[i] > 0.3 and ecg[i-1] < ecg[i] > ecg[i+1]:
            peaks.append(i)
            i += 100
        else:
            i += 1
    return np.array(peaks)

def getIntervals(ecg):
            """Вычисление интервалов между R-зубцами"""
            peaks = getPeaks(ecg)
            intervals = np.zeros(len(peaks)-1)
            for i in range(len(intervals)):
                intervals[i] = peaks[i+1] - peaks[i]
            return intervals


def getHR(ecg):
        """Вычисление ЧСС по индексам пиков. Вычисляется по крайним
        пикам в подаваемом в качестве аргумента сигнале"""
        peaks = getPeaks(ecg)
        hr = 60/((peaks[-1] - peaks[1])/hz/(len(peaks)-1))
        return hr

# Сохраним сигнал в отдельный массив
ecg = traindf['ecg'].values
# визуализируем участок сигнала
plt.figure(figsize=(15,10))
plt.plot(ecg[:500])

# Построим график зависимости ЧСС от номера отрезка данных. Участки,
# соответствующие состоянию со сниженным уровнем внимания отметим
# красным. Как видно, в таки моменты мы наблюдаем наиболее низкую ЧСС.
hr = pd.DataFrame({'hr': [getHR(ecg[i*15*hz:(i+1)*15*hz])
    for i in range(len(ecg)//(15*hz))],'target': [tar for tar in traindf.target.values[::15*hz]]})
plt.figure(figsize=(15,10))
plt.plot(hr.hr[hr.target==1])
plt.plot(hr.hr[hr.target==0], 'r')
