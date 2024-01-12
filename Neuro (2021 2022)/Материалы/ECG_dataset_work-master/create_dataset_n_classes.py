# -*- coding: utf-8 -*
#
# создать датасет: на вход - n диагнозов, на выход датасет вида [x, y] где len(y[i]) = n
import os
import numpy as np
import pickle as pkl
import pyedflib

from raw_dataset_to_pandas_frame import (
    get_pandas_dataframe, get_n_most_freq_names
)

def _get_signal_from_file(filename_edf):
    """

    :param filename_edf: полное имя .edf файла
    :return: сигнал с нулевого отведения (т.е. 1 из 12 штук, в форме вектора)
    """
    try:
        f = pyedflib.EdfReader(filename_edf)
        n = f.signals_in_file
        signals = np.zeros((n, f.getNSamples()[0]))
        for i in np.arange(n):
            signals[i, :] = f.readSignal(i)
        f._close()
        del f
        return signals
    except Exception as  exeption:
        return None

def generate_dataset(list_of_diagnoses, folder_with_raw_dataset, name_pkl):
    """
    генерит датасет, который можно непосредственно скормить нейросети
    :param list_of_diagnoses: список ключей-диагнозов, которые сеть будет учиться диагностировать
    :param folder_with_raw_dataset: папочка с edf/json файлами
    :param name_pkl: имя создаваемого файла с датасетом, должно иметь расширение .pkl
    :param is_2d: сохранять иксы как векторы или как двухмерные картинки (если = True, то как картинки)
    :return: незапакованный в файл датасет
    """
    print("составляем датасет ["+ name_pkl + "] по сдедующему списку диагнозов: " + str(list_of_diagnoses))
    df = get_pandas_dataframe(folder_with_raw_dataset)
    x = []
    y = []
    errors = []
    for index, row in df.iterrows():
        edf_file = row['edf_file']
        signal = _get_signal_from_file(edf_file)
        if signal is None:
            errors.append(edf_file)
            continue
        x.append(signal)
        diagnoz_vector = []
        for diagnoz_name in list_of_diagnoses:
            diagnoz_vector.append(row[diagnoz_name])
        y.append(diagnoz_vector)

    assert len(x) == len(y)

    dict = {'x': x, 'y': y, 'summary': 'датасет n-мерного диагноза - электро-ось'}
    outfile = open(name_pkl, 'wb')
    pkl.dump(dict, outfile)
    outfile.close()
    print("датасет сохранен в файл, количество записей = " + str(len(x)))
    if (len(errors)!=0):
        print ("не удалось добавить в датасет файлы: " + str(errors))
    return x, y


if __name__ == "__main__":
    folder_with_raw_dataset = 'C:\\ecg_new'  # путь к папке с исходными файлами .edf, .json
    # списки диагнозов копипастить из visualisation.ipnb (т.е. запустить его выполняться и посмотреть, у каких списков/диагнозов хорошая представленность получится)
    #list_of_diagnoses = ['normal']
    list_of_diagnoses = get_n_most_freq_names(n=25)
    name_for_dataset_file = 'TOP_15(new).pkl'

    folder_name = 'all_datasets_here'
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    name_for_dataset_file = os.path.join(folder_name, name_for_dataset_file)
    generate_dataset(list_of_diagnoses, folder_with_raw_dataset, name_for_dataset_file)



