# -*- coding: utf-8 -*

#  получаем имя файла (dset_1d_pkl) с 1д датасетом, генерим соотвествующий ему
# 2d датасет и сохраняем его в файл dset_name_2d, параллельно сохраняя
# картинки в new_folder

import os
import shutil
import pickle as pkl
import matplotlib.pyplot as plt
import easygui
import cv2

# https://matplotlib.org/users/image_tutorial.html


def gen_2d_dset(dset_1d_pkl, dset_name_2d):
    # часть 1 - загружаем исходный датасет из файла
    print("загружаем датасет векторов из файла " + dset_1d_pkl)
    infile = open(dset_1d_pkl, 'rb')
    new_dict = pkl.load(infile)
    infile.close()
    print("первичный датасет загружен, в нем найдено записей = : " + str(len(new_dict['x'])))

    # часть 2 - генерим картинки
    x = new_dict['x']
    temp_folder = 'temp'
    new_dict['x'] = _generate_pictures(x, temp_folder)
    shutil.rmtree('temp')  # удаляем папку с промежуточным слжебым мусором

    # часть 3 - сохраняем новый датасет в файл
    outfile = open(dset_name_2d, 'wb')
    pkl.dump(new_dict, outfile)
    outfile.close()
    print(" сохранеили датасет из картинок в файл: " + dset_name_2d)



def _generate_pictures(x_1d, temp_folder):
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder)

    print(" генерируются картинки...")
    pics = []
    id = 0
    for x in x_1d:
        pic_name = str(id)+".png"
        pic_name = os.path.join(temp_folder,pic_name)
        _save_signal_as_pic(x, pic_name)
        picture= cv2.imread(pic_name, cv2.COLOR_RGB2GRAY)
        pics.append(picture)
        id+=1
    print("2d imgs created...")

    return pics

def _save_signal_as_pic(x, name):
    fig = plt.figure()
    ax = fig.gca()

    ax.plot(x)
    ax.axis('off')
    #ax.set_ylim(bottom=ymin, top=ymax) TODO
    fig.savefig(name)
    plt.close(fig)


if __name__ == "__main__":
    filename_pkl = easygui.fileopenbox()
    dset_name_2d = "ELECTRIC_AXIS_2d.pkl"


    folder_name = 'all_datasets_here'
    dset_name_2d = os.path.join(folder_name, dset_name_2d)

    gen_2d_dset(filename_pkl, dset_name_2d)