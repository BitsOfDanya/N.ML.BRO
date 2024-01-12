#генеим маску разметку и сохраняем датасет ввиде: x, ann

import os
import json
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import pyedflib
import numpy as np
import pickle as pkl

path_to_file = "./delineation.json"
folder_with_raw_dataset = 'C:\\ecg_new'

# Порядок отведений не менять!! он соответствует 1-12
leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
for i in range(len(leads)):
    leads[i]= 'lead_'+ leads[i]

tables_names = ['qrs', 't', 'p']
MAX=5000

def generate_mask_for_ecg_lead_by_one_table(data, raw_ecg_id, lead_name, table_name):
    mask =[0]*MAX #забиваем сначала маску нулями
    qrs_table = data[raw_ecg_id][lead_name][table_name]
    for triplet in qrs_table:
        start = triplet[0]
        end = triplet[2]
        for i in range(start, end,1):
            mask[i] = 1
    return mask

def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]

def draw_masks_for_ecg_lead(masks, masks_names, ecg_case, lead_name):
    signal = get_ecg_by_lead_name(ecg_case, lead_name)
    if signal is None:
        return
    figname = ecg_case+"_"+lead_name+".png"
    plt.plot(signal)

    for mask in masks:
        mask_with_nons = zero_to_nan(mask)# не будем рисовать зануленную часть маски
        plt.plot(mask_with_nons)

    plt.title('mask visualisation')
    plt.ylabel('masks and ecg')
    plt.xlabel('time')
    plt.legend(['signal']+masks_names, loc='upper left')
    plt.savefig(figname)


def get_ecg_by_lead_name(ecg_case, lead_name):
    filename_edf = os.path.join(folder_with_raw_dataset, ecg_case + ".edf")
    channel_index = leads.index(lead_name)
    try:
        f = pyedflib.EdfReader(filename_edf)
        n = f.signals_in_file
        assert n >= channel_index
        signal = f.readSignal(channel_index)
        f._close()
        del f
        return signal
    except Exception as  exeption:
        print("какая-то фигня с файлом кардиограммы " + filename_edf)
        return None

def _get_signal_from_file(ecg_case):
    """
    :param filename_edf: полное имя .edf файла
    :return: сигнал с нулевого отведения (т.е. 1 из 12 штук, в форме вектора)
    """
    filename_edf = os.path.join(folder_with_raw_dataset, ecg_case + ".edf")
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

def print_data(data):
    print(data.keys())  # айдишники пациентов
    print(data['2731'].keys()) # айдишники отведений
    print(data['2731']['lead_v2'].keys()) #айдишники триплетов

def handle_data(data):
    cases = 0
    for file in os.listdir(folder_with_raw_dataset):
        filename = os.fsdecode(file)
        if filename.endswith(".edf"):
            case_id = filename[0:-len(".edf")]
            if case_id in data.keys():
                cases +=1
    print ("нашлись аннотации для " + str(cases) + " пациентов нашего датасета " + folder_with_raw_dataset)

def generate_masks_for_ecg_lead(data, ecg_case, lead_name):
    masks = []
    for table_name in tables_names:
        mask = generate_mask_for_ecg_lead_by_one_table(data,
                                                   raw_ecg_id=ecg_case,
                                                   lead_name=lead_name,
                                                   table_name=table_name)
        masks.append(mask)
    return masks, tables_names

def test(data):
    # рисуем маски для первых 5 пациентов для первого отведения - рисунки сохраняем в файлы
    i = 0
    lead_name = leads[0]
    for ecg_case in data.keys():
        i+=1
        masks, masks_names = generate_masks_for_ecg_lead(data, ecg_case, lead_name)
        draw_masks_for_ecg_lead(masks=masks,
                            masks_names=masks_names,
                            ecg_case=ecg_case,
                            lead_name=lead_name)
        plt.clf()
        if i>5:
            break

def generate_dataset_argentina(data):
    #x = 12 отведений ЭКГ, ann = трехслояная маска первого отведения
    print("начинаем генерацию датасета аннотаций,  x = 12 отведений ЭКГ, ann = трехслояная маска первого отведения")
    dataset_name = "DSET_argentina.pkl"
    lead_name = leads[0]
    x = []
    ann = []
    for ecg_case in data.keys():
        masks, masks_names = generate_masks_for_ecg_lead(data, ecg_case, lead_name)
        signal = _get_signal_from_file(ecg_case)
        if signal is None:
            continue
        assert len(masks[0]) == len(signal[0]) == MAX
        x.append(signal)
        ann.append(np.array(masks))
    dict = {'x': np.array(x),
            'ann': np.array(ann),
            'summary':'3 binary masks'}
    outfile = open(dataset_name, 'wb')
    pkl.dump(dict, outfile)
    outfile.close()
    print("датасет сохранен в файл " + dataset_name)
    print("характриситки датасета: инпут = "+ str(dict['x'].shape) + ", аутпут = "+ str(dict['ann'].shape))

def generate_masks_for_ecg_several_leads(data, ecg_case, leads_names):
    masks = []
    masks_names = []
    for lead_name in leads_names:
        masks_one_lead, masks_names_one_lead = generate_masks_for_ecg_lead(data, ecg_case, lead_name)
        for mask in masks_one_lead:
            masks.append(mask)
        masks_names.append(masks_names_one_lead)
    return masks, masks_names



def generate_dataset_jamayka(data):
    #x = 12 отведений ЭКГ, ann = 12*3 штук бинарных масок
    print("начинаем генерацию датасета аннотаций,  x = 12 отведений ЭКГ, ann = 12*3 штук бинарных масок")
    dataset_name = "DSET_jamayka.pkl"

    x = []
    ann = []
    for ecg_case in data.keys():
        masks, masks_names = generate_masks_for_ecg_several_leads(data, ecg_case, leads_names=leads)
        signal = _get_signal_from_file(ecg_case)
        if signal is None:
            continue
        assert len(masks[0]) == len(signal[0]) == MAX
        x.append(signal)
        ann.append(np.array(masks))
    dict = {'x': np.array(x),
            'ann': np.array(ann),
            'summary':'3*12 binary masks'}
    outfile = open(dataset_name, 'wb')
    pkl.dump(dict, outfile)
    outfile.close()
    print("датасет сохранен в файл " + dataset_name)
    print("характриситки датасета: инпут = "+ str(dict['x'].shape) + ", аутпут = "+ str(dict['ann'].shape))

def get_mixed_mask(mask1,mask2,mask3):
    mixed_mask = mask1 + 2*mask2 + 3*mask3
    return mixed_mask

def generate_mask_for_ecg_lead_NIXED(data, ecg_case, lead_name):
    masks = []
    for table_name in tables_names:
        mask = generate_mask_for_ecg_lead_by_one_table(data,
                                                       raw_ecg_id=ecg_case,
                                                       lead_name=lead_name,
                                                       table_name=table_name)
        masks.append(mask)
    masks = np.array(masks)
    return get_mixed_mask(masks[0], masks[1], masks[2])

def generate_masks_for_ecg_several_leads_MIXED(data, ecg_case, leads_names):
    masks = []
    for lead_name in leads_names:
        maska_one_lead = generate_mask_for_ecg_lead_NIXED(data, ecg_case, lead_name)
        masks.append(maska_one_lead)
    return masks

def generate_dataset_urugvay(data):
    #x = 12 отведений ЭКГ, ann = 1 смешанаая маска первого отведения
    print("начинаем генерацию датасета аннотаций,  x = 12 отведений ЭКГ, ann = 1 смешанаая маска первого отведения")
    dataset_name = "DSET_urugvay.pkl"
    lead_name = leads[0]
    x = []
    ann = []
    for ecg_case in data.keys():
        maska = generate_mask_for_ecg_lead_NIXED(data, ecg_case, lead_name)
        signal = _get_signal_from_file(ecg_case)
        if signal is None:
            continue

        assert len(maska) == len(signal[0]) == MAX
        x.append(signal)
        ann.append(np.array(maska))
    dict = {'x': np.array(x),
            'ann': np.array(ann),
            'summary':'1 not-binary maska'}
    outfile = open(dataset_name, 'wb')
    pkl.dump(dict, outfile)
    outfile.close()
    print("датасет сохранен в файл " + dataset_name)
    print("характриситки датасета: инпут = "+ str(dict['x'].shape) + ", аутпут = "+ str(dict['ann'].shape))

def generate_dataset_ostrov_paskhy(data):
    #x = 12 отведений ЭКГ, ann = 12 небинарных масок (каждому отведению по одной смешаной маске)
    print("начинаем генерацию датасета аннотаций,  x = 12 отведений ЭКГ, ann = 12 штук НЕбинарных масок")
    dataset_name = "DSET_ostrov_paskhy.pkl"

    x = []
    ann = []
    for ecg_case in data.keys():
        masks = generate_masks_for_ecg_several_leads_MIXED(data, ecg_case, leads_names=leads)
        signal = _get_signal_from_file(ecg_case)
        if signal is None:
            continue
        assert len(masks[0]) == len(signal[0]) == MAX
        x.append(signal)
        ann.append(np.array(masks))
    dict = {'x': np.array(x),
            'ann': np.array(ann),
            'summary':'3*12 binary masks'}
    outfile = open(dataset_name, 'wb')
    pkl.dump(dict, outfile)
    outfile.close()
    print("датасет сохранен в файл " + dataset_name)
    print("характриситки датасета: инпут = "+ str(dict['x'].shape) + ", аутпут = "+ str(dict['ann'].shape))

with open(path_to_file, 'r') as f:
    data = json.load(f)
    #handle_data(data) распечатать стр-ру джсон - файла с аннотациями
    #test(data) нарисовать несколько картинок экг с аннотациями
    #generate_dataset_argentina(data)
    #generate_dataset_jamayka(data)
    #generate_dataset_urugvay(data)
    generate_dataset_ostrov_paskhy(data)


