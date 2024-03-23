#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as ppc

raw_data = np.loadtxt( 'photo_diode.dat', delimiter='\t', dtype=np.int) 
# в raw_data находятся данные с сенсоров (более подробно см. описание к BiTronics Studio EEG edition)
# столбцы 0-7 - данные с датчиков ЭЭГ, столбец 8 - данные с фотодиода

ph_diode = np.transpose(raw_data)[8] # выделяем данные с фотодиода в отдельный массив

ph_scaled = ppc.scale(ph_diode)
# scale нормирует данные от фотодиода, после этого их среднее значение становится равно нулю. 

ph_clean = np.where( ph_scaled > 0, 1, 0) 
# where с такими параметрами все отрицательные значения делает нулевыми, а все положительные - единицей, 
# т.е. теперь у нас в массиве ph_clean моменты со включенной подсветкой обозначены 1, а с выключенной - 0   

ph_min = ph_clean[1:-1] - ph_clean[2:] # вычитаем массив сам из себя со сдвигом на 1 точку, теперь у нас момент включения подсветки обозначается '1', а момент выключения '-1', все остальные точки нулевые

starts_point = np.where(ph_min  > 0)[0] # получаем массив, в котором лежат индексы точек, в которых начинается подсветка строки либо столбца

# отобразим все этапы обработки данных на графике
fig, axs = plt.subplots(4)
axs[0].plot(ph_diode, color='tab:green', label='исходные данные')
axs[1].plot(ph_scaled, color='tab:orange', label='после scale')
axs[2].plot(ph_clean, color='tab:blue', label='после where')
axs[3].plot(ph_min, color='tab:red', label='после вычитания со сдвигом')
for ax in axs:
    ax.legend(loc='lower left', shadow=True, fontsize='x-large')
plt.show()
#=============================================================================================
