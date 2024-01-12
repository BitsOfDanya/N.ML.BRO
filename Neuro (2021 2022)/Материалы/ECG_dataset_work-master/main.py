# -*- coding: utf-8 -*
from pprint import pprint

import easygui
import pickle as pkl
def open_dataset():
    filename_pkl = easygui.fileopenbox("выберите файл с датасетом")
    print("загружаем 2д-датасет из файла " + filename_pkl)
    infile = open(filename_pkl, 'rb')
    new_dict = pkl.load(infile)
    infile.close()
    assert len(new_dict['x']) == len(new_dict['y'])
    print("загрузили, нашли записей в нем : " + str(len(new_dict['x'])))
    return new_dict['x'], new_dict['y']

if __name__ == "__main__":
    x, y = open_dataset()
    pprint(y)
    pprint(x)