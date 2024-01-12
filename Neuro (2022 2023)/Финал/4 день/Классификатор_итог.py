import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from operator import itemgetter
from time import sleep
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

models=[]
models.append(DecisionTreeClassifier())
models.append(GaussianNB())


def spectrum(y, Fs=256):
    # частота дискретизации
    n = len(y)  # длинна отрезка
    k = np.arange(n)
    T = n / Fs
    frq = k / T
    frq = frq[range(n // 2)]
    Y = np.fft.fft(y) / n
    Y = Y[range(n // 2)]
    return frq, Y


# In[функция для обучения и проверки классификаторов]:
def neur():
    global X, y
    # разделение масивов и перемешивание для обучения и тестирования
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=50, shuffle=True)
    res = []

    # перебирание моделей классификаторов
    for model in models:
        try:
            print(model)
            model.fit(X_train, y_train)
            # тестирование
            kfold = StratifiedKFold(n_splits=2, random_state=2, shuffle=True)
            cv_results = cross_val_score(model, X_test, y_test, cv=kfold, scoring='balanced_accuracy')
            res.append([str(model), ('%s: %f (%f)' % ("", cv_results.mean(), cv_results.std())), cv_results])
            print("Good")
        except BaseException as e:
            print(e)
    sleep(1)

    for i in res:
        print(i[0])
    # сортировака лучших классефикаторов
    res = sorted(res, key=itemgetter(1))
    return list(map(lambda x: x[0: 2], res))


# In[чтение данных после эксперемента]:
with open('AlexBRL.csv', "r") as f:
    brl = np.rot90(np.array([list(map(float, i)) for i in list(csv.reader(f, delimiter=';'))]), -1)[1:]
with open('AlexEEG.csv', "r") as f:
    eeg = np.rot90(np.array([list(map(float, i)) for i in list(csv.reader(f, delimiter=';'))]), -1)[1:]
with open('AlexEKG.csv', "r") as f:
    fpg = np.rot90(np.array([list(map(float, i)) for i in list(csv.reader(f, delimiter=';'))]), -1)[1:]
with open('AlexKGR.csv', "r") as f:
    kgr = np.rot90(np.array([list(map(float, i)) for i in list(csv.reader(f, delimiter=';'))]), -1)[1:]

# In[преобразование дынных в масивы X и Y]:
Х = []
y = eeg[1]
for i in range(len(eeg[0])):
    Х.append([brl[0][i], eeg[0][i], fpg[0][i], kgr[0][i]])

X = np.array(Х, dtype=np.float32)

# In[Использование Моделей с функции neur]:
mud = neur()
print(*mud, sep=" ")  # вывод классефикаторов и результаты

np.savetxt('test_1.txt', y)
print(", ".join(map(str, y)))
