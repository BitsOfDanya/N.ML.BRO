# -*- coding: utf-8 -*

# на вход подаем папку с сырым датасетом, на выходе получаем пандас датафрейм, одним
# из толбцов будет название айдишкник случая, другим полным путь к едф-файлу случая,
# и остальные 84 - заполненные поля диагноза случая
import os
import json
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

def _get_entries_as_list_of_dicts(folder):
    """
    обходим все едф-ки, для каждой ищем соответсвующий ей json и потрошим его
    :param folder: путь к папке c сырым датасетом
    :return: список словарей, где каждый словарь соотв. пациенту
    """
    entries = []
    errors = []
    print("строим табличку pandas по коллекции файлов из " + folder)
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(".edf"):
            entry = {}
            case_id = filename[0:-len(".edf")]
            json_filename = case_id + ".json"
            json_filename = os.path.join(folder, json_filename)
            if os.path.isfile(json_filename):
                entry["case_id"]=case_id
                entry["edf_file"]=os.path.join(folder, filename)
                with open(json_filename) as json_file:
                    json_dict = json.load(json_file)
                    # при записи дигнозов в таблицу сделаем их 0-1, а не тру-фолс
                    for diagnosis in json_dict.keys():
                        if json_dict[diagnosis] is True: entry[diagnosis] = 1
                        else: entry[diagnosis] = 0
                    entries.append(entry)
            else:
                errors.append(case_id)  # не нашлось джсона, соответствующегго этому эдф
    if len(errors) != 0:
        print("не нашли диагнозов к некотрым экг: " + str(errors))
    return entries


def get_pandas_dataframe(folder):
    list_of_dicts = _get_entries_as_list_of_dicts(folder)
    print ("в папке " + folder + " найдено пациентов: " + str(len(list_of_dicts)))
    print("(файлы .edf на валидность не проверялись)")
    dframe = pd.DataFrame(list_of_dicts)
    return dframe

def get_n_most_freq_names(n, folder_with_files= 'C:\\ecg_new' ):
    df = get_pandas_dataframe(folder_with_files)

    df.loc['Total'] = df.sum()

    df = df.T

    df = df.iloc[:, [-1]]
    df = df[pd.to_numeric(df['Total'], errors='coerce').notnull()]
    df = df.sort_values(by=['Total']).tail(n)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    names = list(df.index.values)
    df.plot.bar()
    plt.savefig("bar_plot_distrubutioon.png")
    return names

if __name__ == "__main__":
    folder_with_files = 'C:\\ecg'  # путь к папке с исходными файлами .edf, .json
    df = get_pandas_dataframe(folder_with_files)
    print(tabulate(df, headers='keys', tablefmt='psql'))

    print(get_n_most_freq_names(n=10))



