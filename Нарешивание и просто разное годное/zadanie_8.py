#coding:utf-8
import numpy as np
from sklearn import preprocessing as ppc
from scipy import signal as sig
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import discriminant_analysis as LDA

# функция для предобработки данных 
# варьируя разные методы предобработки можно существенно улучшить либо ухудшить результаты обучения и распознавания
# DATA - исходный массив данных
# epoStart - массив, в котором содержатся индексы начала каждой "эпохи", т.е. отрезка времени, в котором может появиться (или не появиться) P300
def epoProc(DATA, epoStart):
    # вырезаем часть данных, в которых можно ожидать появление Р300, излишние данные могут приводить к дополнительным ошибкам в распознавании 
    Ep = DATA[epoStart:epoStart+300, :6]
    # уменьшаем количество точек, т.к. пик Р300 достаточно длительный по времени и большое количество точек могут приводить к дополнительным ошибкам в распознавании 
    ep = sig.decimate(Ep, 5, ftype='fir', axis=0)
    # вычитаем "базовую линию", чтобы данные, в которых нет Р300 всегда были в среднем около нуля 
    ep -= np.mean(ep[0:20])
    ep = ep.reshape(ep.shape[0]*ep.shape[1])
    return ep
#=============================================================================================

# функция для загрузки данных из файла и нахождения точек, в которых включается подсветка, по данным с фотодиода 
# fname - имя .dat и .txt файлов, сохранённых программой BiTronics Studio EEG edition (более подробно см. описание к программе)
def preload_data(fname):
    data_fname = fname + '.dat'
    stamps_fname = fname + '.txt'

    blinks_data = np.loadtxt(stamps_fname, delimiter='\t', dtype=np.int)
    # в blinks_data записан порядок подсветки алфавита в окне программы (более подробно см. описание к BiTronics Studio EEG edition)
    #столбец 1 - номер подсвеченной строки или столбца, начиная с нуля
    #столбец 2 - если значение '0' то была подсвечена строка, если '1' - то столбец

    raw_data = np.loadtxt(data_fname, delimiter='\t', dtype=np.int) 
    #в raw_data находятся данные с сенсоров (более подробно см. описание к BiTronics Studio EEG edition)
    # столбцы 0-7 - данные с датчиков ЭЭГ, столбец 8 - данные с фотодиода, все с частотой 240Гц
    ph_diode = np.transpose(raw_data)[8] # выделяем данные с фотодиода в отдельный массив
    ph_clean = np.where( ppc.scale(ph_diode) > 0, 1, 0) 
    # scale нормирует данные от фотодиода, после этого их среднее значение становится равно нулю. 
    # where все отрицательные значения делает нулевыми, а все положительные - единицей, 
    # т.е. теперь у нас в массиве ph_clean моменты со включенной подсветкой обозначены 1, а с выключенной - 0   
    ph_min = ph_clean[1:-1] - ph_clean[2:] # теперь у нас момент включения подсветки обозначается '1', а момент выключения '-1', все остальные точки нулевые
    starts_point = np.where(ph_min  > 0)[0] # получаем массив, в котором лежат индексы точек, в которых начинается подсветка строки либо столбца
    # далее убираем из массива все индексы, после которых в массиве данных нет хотя-бы 300 точек, т.к. мы работаем с участками данных ("длиной эпохи") 300 точек 
    while (starts_point[-1] + 300) > raw_data.shape[0]:
        starts_point = np.delete(starts_point, -1)
    
    assert len( starts_point ) <= len(blinks_data), 'Blinks count error'# проверяем, что распознанное количество вспышек не больше реального, чего, очевидно, не может быть
    return raw_data, blinks_data, starts_point
#=============================================================================================


# функция, которая загружает данные, разделяет их на данные от строк и столбцов и подготавливает их к распознаванию обученными алгоритмами
# fname - имя .dat и .txt файлов, сохранённых программой BiTronics Studio EEG edition (более подробно см. описание к программе)
def get_data( fname ):
    # загружаем данные из файла
    raw_data, blinks_data, starts_point = preload_data(fname)
    starts_row = np.empty(0, dtype=int) # в этом массиве будут индексы из starts_point, соответствующие подсветке строк 
    starts_col = np.empty(0, dtype=int) # в этом массиве будут индексы из starts_point, соответствующие подсветке столбцов
    num_row = np.empty(0, dtype=int) # в этом массиве будут номера строк, согласно тому в какой последовательности подсвечивались строки
    num_col = np.empty(0, dtype=int) # в этом массиве будут номера строк, согласно тому в какой последовательности подсвечивались столбцы

    # в соответствии со значением blinks_data[ii][2], распределяем данные по массивам отдельно для строк отдельно для столбцов 
    for ii in range( len( starts_point ) ):
        if blinks_data[ii][2] == 0: # если blinks_data[ii][2] == 0, значит была подсвечена строка; если == 1 - то столбец
            starts_row = np.append(starts_row, starts_point[ii])
            num_row = np.append(num_row, blinks_data[ii][1])
        else:
            starts_col = np.append(starts_col, starts_point[ii])
            num_col = np.append(num_col, blinks_data[ii][1])
    # собираем массивы из предобработанных данных
    data_col = np.array([epoProc(raw_data, i) for i in starts_col])
    data_row = np.array([epoProc(raw_data, i) for i in starts_row])
    # шкалируем данные - среднее по каждому каналу данных сдвигается к нулю, а амплитуды сжимаются так, чтобы среднеквадратичное отклонение стало равно единице
    data_col = ppc.scale(data_col)
    data_row = ppc.scale(data_row)
    return data_row, num_row, data_col, num_col 
#=============================================================================================

# функция, которая загружает данные, разделяет их на данные с событиями Р300 и без них, для дальнейшего обучения алгоритмов распознавания
# fname - должно совпадать с буквой, которую загадывал оператор. Имя .dat и .txt файлов, сохранённых программой BiTronics Studio EEG edition (более подробно см. описание к программе)
def get_train_data( fname ):
    # загружаем данные из файла
    raw_data, blinks_data, starts_point = preload_data(fname)
    letter = fname
    alphabet = {
        'А':(0,0), 'Б':(1,0), 'В':(2,0), 'Г':(3,0), 'Д':(4,0), 'Е':(5,0), 'Ж':(6,0),
        'З':(0,1), 'И':(1,1), 'Й':(2,1), 'К':(3,1), 'Л':(4,1), 'М':(5,1), 'Н':(6,1),
        'О':(0,2), 'П':(1,2), 'Р':(2,2), 'С':(3,2), 'Т':(4,2), 'У':(5,2), 'Ф':(6,2),
        'Х':(0,3), 'Ц':(1,3), 'Ч':(2,3), 'Ш':(3,3), 'Щ':(4,3), 'Ы':(5,3), 'Ь':(6,3),
        'Э':(0,4), 'Ю':(1,4), 'Я':(2,4), '0':(3,4), '1':(4,4), '2':(5,4), '3':(6,4),
        '4':(0,5), '5':(1,5), '6':(2,5), '7':(3,5), '8':(4,5), '9':(5,5) }
    # находим строку и столбец, в которых находится ожидаемая оператором буква. При подсветке этой строки и столбца в данных должен появляться Р300
    col, row = alphabet[letter]
    start_P300 = np.empty(0, dtype=int)
    start_nonP300 = np.empty(0, dtype=int)
    # распределяем индексы из starts_point по массивам: с Р300 и без
    for ii in range( len( starts_point ) - 1 ):
        if ( (blinks_data[ii][1] == row) and (blinks_data[ii][2] == 0) ) or ( (blinks_data[ii][1] == col) and (blinks_data[ii][2] == 1) ): # blinks_data[ii][2] == 0 - raw, == 1 -column
            start_P300 = np.append(start_P300, starts_point[ii])
        else:
            start_nonP300 = np.append(start_nonP300, starts_point[ii])
    
    Target = np.array([epoProc(raw_data, i) for i in start_P300])
    NonTarget = np.array([epoProc(raw_data, i) for i in start_nonP300])
    X = ppc.scale(np.vstack((Target, NonTarget)))
    y = np.hstack((np.ones(Target.shape[0]), np.zeros(NonTarget.shape[0])))
    
    return X, y
#=============================================================================================
# функция которая выводит процент совпадения двух массивов
def perc2( predicted, y_test ):
    return np.mean(predicted == y_test) *100.0
#=============================================================================================
# загружаем для обучения несколько массивов данных
X0, y0 = get_train_data('0')
X, y = get_train_data('7')
X0= np.concatenate((X0, X), axis = 0);
y0= np.concatenate((y0, y), axis = 0);
X, y = get_train_data('8')
X0= np.concatenate((X0, X), axis = 0);
y0= np.concatenate((y0, y), axis = 0);
X, y = get_train_data('Ж')
#X0= np.concatenate((X0, X), axis = 0);
#y0= np.concatenate((y0, y), axis = 0);
#X, y = get_train_data('Й')
#X0= np.concatenate((X0, X), axis = 0);
#y0= np.concatenate((y0, y), axis = 0);
#X, y = get_train_data('К')
#X0= np.concatenate((X0, X), axis = 0);
#y0= np.concatenate((y0, y), axis = 0);
X, y = get_train_data('М')
X0= np.concatenate((X0, X), axis = 0);
y0= np.concatenate((y0, y), axis = 0);
X, y = get_train_data('С')
X0= np.concatenate((X0, X), axis = 0);
y0= np.concatenate((y0, y), axis = 0);

# разделяем набор обучающих данных на обучающий и тестовый массивы.
X_train, X_test, y_train, y_test = train_test_split( X0, y0, test_size=0.1, shuffle=True)
#=============================================================================================

# попробуем создать несколько классификаторов, чтобы выяснить какой в нашем случае лучше подходит
# для лучшего распознавания можно попробовать менять настройки классификаторов, а также попробовать другие типы классификаторов
clf = svm.SVC(probability=True)
lr = LogisticRegression()
cls = LDA.LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage='auto')

# обучаем классификатор
clf.fit(X_train, y_train)
# пытаемся распознать тестовые данные
# predict дает нам бинарный ответ - являются ли предложенные данные тем, что мы ищем или нет
predicted = clf.predict(X_test)
# сообщаем пользователю, насколько успешным было распознавание
print('SVC predict percentage: ', perc( predicted, y_test ) )


# повторяем то-же самое c другим классификатором
lr.fit(X_train, y_train)
predicted = lr.predict(X_test)
print('LogisticRegression predict percentage: ', perc( predicted, y_test ))

# и ещё раз с третьим классификатором, чтобы по результатам теста выбрать наилучший для имеющегося набора данных
cls.fit(X0, y0)
predicted = cls.predict(X_test)
print('LDA predict percentage: ', perc( predicted, y_test ),'\n')

# загружаем данные с неизвестной буквой для её поиска
data_row, nums_row, data_col, nums_col = get_data('Й')

alphabet = [
    ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж'],
    ['З', 'И', 'Й', 'К', 'Л', 'М', 'Н'],
    ['О', 'П', 'Р', 'С', 'Т', 'У', 'Ф'],
    ['Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ы', 'Ь'],
    ['Э', 'Ю', 'Я', '0', '1', '2', '3'],
    ['4', '5', '6', '7', '8', '9', '?'] ]

# пытаемся распознать тестовые данные
# predict_proba дает нам вероятность того, что предложенные данные являются тем, что мы ищем
res_row = lr.predict_proba(data_row)
# мы получили вероятности для каждой из подсвеченных строк и ищем ту, для которой вероятность наличия Р300 наивысшая
# в результате получаем индекс искомой строки в массиве res_row и по индексу можем найти номер строки в массиве nums_row
num_row = np.where( res_row[:,1] == np.amax( res_row[:,1]) )[0][0]
# повторяем то-же самое для колонок
res_col = lr.predict_proba(data_col)
num_col = np.where( res_col[:,1] == np.amax( res_col[:,1]) )[0][0]
# для получения более точного результата можно подсвечивать требуемую букву несколько раз, суммировать полученные вероятности для каждой строки и столбца и выбирать наилучшее совпадение уже после этого 

# выводим пользователю полученные индексы для массивов res_row и res_col
print('SVC predicted col and row: ',num_row, num_col)
# и соответствующую букву
print('''predicted symbol: "''', alphabet[nums_row[num_row]][ nums_col[num_col]],'''"''')

# повторяем то-же самое c другим классификатором
res_row = clf.predict_proba(data_row)
num_row = np.where( res_row[:,1] == np.amax( res_row[:,1]) )[0][0]
res_col = clf.predict_proba(data_col)
num_col = np.where( res_col[:,1] == np.amax( res_col[:,1]) )[0][0]
print('LogisticRegression predicted col and row: ',num_row, num_col)
print('''predicted symbol: "''', alphabet[nums_row[num_row]][ nums_col[num_col]],'''"''')

# и ещё раз с третьим классификатором
res_row = cls.predict_proba(data_row)
num_row = np.where( res_row[:,1] == np.amax( res_row[:,1]) )[0][0]
res_col = cls.predict_proba(data_col)
num_col = np.where( res_col[:,1] == np.amax( res_col[:,1]) )[0][0]
print('LDA predicted col and row: ',num_row, num_col)
print('''predicted symbol: "''', alphabet[nums_row[num_row]][ nums_col[num_col]],'''"''')

# видно, что хотя буква может быть определена верно всеми тремя классификаторами, в зависимости от применяемого классификатора полученные индексы буквы в исходном массиве данных могут быть разными 

