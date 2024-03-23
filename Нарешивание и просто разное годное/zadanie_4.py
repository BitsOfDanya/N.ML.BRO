#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

# загружаем данные из файла
D = np.loadtxt('P300_dataSet.txt')

# находим индексы, в которых Label меняется с 0 на 1, т.е. ожидается сигнал P300
T_starts = np.nonzero((D[1:, -1] - D[:-1, -1]) == 1)[0]
# рассчитываем, сколько точек появится на заявленной длине эпохи 400мс + 800мс при частоте оцифровки 250Гц
epo_len = int( 250 * ( (800+400)/1000 ) )

T = np.zeros((epo_len, 6))
# Суммируем все участки данных ("эпохи"), в которых ожидается Р300
for t in T_starts:
    T += D[t:t+epo_len, :-4]

T = T / len(T_starts)

# строим полученные суммарные кривые
#handle = plt.plot(lab[1]['index'], lab[1].drop('index', axis=1))
handle = plt.plot(T)
plt.legend(handles=handle, labels=['Cz', 'Pz', 'PO7', 'PO8', 'O1', 'O2'], loc=1)
plt.ylabel('mV')
plt.xlabel('ms')
plt.show()

