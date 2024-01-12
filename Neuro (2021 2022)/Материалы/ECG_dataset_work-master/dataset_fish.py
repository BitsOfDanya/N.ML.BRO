# для новой сземы обучения (генералиатор- дискриминатор)

# мы генерим датасет здоровых (т.е. электрооось = нормал и ристм нормосистолический)
# потом генерим датасет больных одной какой-то болезнью
# при этом внутри экземпляров одной болезни педсатвлены ОДНОВРЕМЕННЫЕ с ней болезни!
# поэтому внутри датасета больных болезнью посортируем больных по кол-ву одновременных болезней (сумма единиц

import os
from tabulate import tabulate
import pickle as pkl

from raw_dataset_to_pandas_frame import (
    get_pandas_dataframe, get_n_most_freq_names
)

from create_dataset_n_classes import (
    _get_signal_from_file
)

folder_with_raw_dataset = 'C:\\ecg_new'  # путь к папке с исходными файлами .edf, .json
folder_name = 'all_datasets_fish'


def _get_criteria( nessesary_indicators=None, nessesary_zeros=None):
    str_query = ''
    criteria = []
    if nessesary_indicators is not None:
        for diagnos_name in nessesary_indicators:
            criteria.append(diagnos_name + ' == 1')
    if nessesary_zeros is not None:
        for diagnos_name in nessesary_zeros:
            criteria.append(diagnos_name + ' == 0')

    for i in range(len(criteria)):
        str_query = str_query + criteria[i]
        if i < len(criteria) - 1:
            str_query = str_query + " & "
    return str_query

def _append_diversity_column(df, nessesary_indicators):
    if nessesary_indicators is None:
        nessesary_indicators = []
    col_list = list(df)
    for name in nessesary_indicators:
        col_list.remove(name)
    col_list.remove('edf_file')
    col_list.remove('case_id')
    df['diversity'] = df[col_list].sum(axis=1)
    return df

def _get_subframe(nessesary_indicators, nessesary_zeros):
    """
    вернет только те, где nessesary_indicators единицы, а nessesary_zeros нули (остальные как угодно)
    :param nessesary_indicators:
    :param nessesary_zeros:
    :return:
    """
    df = get_pandas_dataframe(folder_with_raw_dataset)

    df = _append_diversity_column(df, nessesary_indicators)
    str_query = _get_criteria(nessesary_indicators=nessesary_indicators,
                              nessesary_zeros=nessesary_zeros)

    df_selected = df.query(str_query)

    # print(tabulate(df_selected, headers='keys', tablefmt='psql'))
    return df_selected

def _get_subframe_ONLY(nessesary_indicators):
    df_selected = _get_subframe(nessesary_indicators, nessesary_zeros=None)

    df_selected = df_selected.query('diversity == 0')
    # print(tabulate(df_selected, headers='keys', tablefmt='psql'))
    return df_selected



def make_dataset(pkl_name, nessesary_indicators=None, zero_indicators=None, mode_only=False):
    x = []
    errors = []
    if mode_only == True:
        df_selected = _get_subframe_ONLY(nessesary_indicators)
    else:
        df_selected = _get_subframe(nessesary_indicators=nessesary_indicators, nessesary_zeros=zero_indicators)
        df_selected = df_selected.sort_values(by=['diversity'], ascending=True) # первыми будут самые чистые
        y = []


    for index, row in df_selected.iterrows():
        edf_file = row['edf_file']
        signal = _get_signal_from_file(edf_file)
        if signal is None:
            errors.append(edf_file)
            continue
        x.append(signal)
        if mode_only is False:
            y.append(row['diversity'])
    if mode_only is False:
        assert len(x) == len(y)
        dict = {'x': x, 'div': y, 'summary': str( nessesary_indicators) + "BUT NOT " + str(zero_indicators)}
    else:
        dict = {'x': x, 'summary': nessesary_indicators}


    outfile = open(pkl_name, 'wb')
    pkl.dump(dict, outfile)
    outfile.close()
    print("датасет сохранен в файл, количество записей = " + str(len(x)))
    if (len(errors) != 0):
        print("не удалось добавить в датасет файлы: " + str(errors))



def make_2_dsets(nessesary_indicators, name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    pkl_name_only = os.path.join(folder_name, "ONLY_"+name)
    pkl_name_not_only = os.path.join(folder_name, name)
    make_dataset(pkl_name_not_only, nessesary_indicators=nessesary_indicators, zero_indicators=None)
    make_dataset(pkl_name_only, nessesary_indicators=nessesary_indicators, mode_only=True)

def make_n_dsets(one_group, second_group, name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    pkl_name_zn = os.path.join(folder_name, "ZN(both)_"+name)
    pkl_name_z_no_n = os.path.join(folder_name, "Z_no_n_" + name)
    pkl_name_n_no_z = os.path.join(folder_name, "N_no_z_" + name)
    pkl_name_no_zn = os.path.join(folder_name, "No_zn(both)_" + name)
    print("!!!---" +str(one_group) + "---NOT: "+ str(second_group))
    make_dataset(nessesary_indicators=one_group, zero_indicators=second_group, pkl_name=pkl_name_z_no_n)

    print("!!!---" + str(second_group) + "---NOT: " + str(one_group))
    make_dataset(nessesary_indicators=second_group, zero_indicators=one_group, pkl_name=pkl_name_n_no_z)

    print("!!!------BOTH: " + str(second_group+one_group))
    make_dataset(nessesary_indicators=second_group+one_group, zero_indicators=None, pkl_name=pkl_name_zn)

    print("!!!------NOT-BOTH: " + str(second_group + one_group))
    make_dataset(nessesary_indicators=None, zero_indicators=second_group + one_group, pkl_name=pkl_name_no_zn)


if __name__ == "__main__":
    name = 'healthy2.pkl'


    nessesary_indicators1 = ['normal', 'regular_normosystole']
    nessesary_indicators2 = ['normal', 'regular_normosystole', 'left_ventricular_hypertrophy']
    nessesary_indicators3 = ['normal', 'regular_normosystole', 'extension_left_atrium']
    nessesary_indicators4 = ['extension_left_atrium', 'left_ventricular_hypertrophy'] #0

    #make_2_dsets(nessesary_indicators1, name)
    make_n_dsets(['normal', 'regular_normosystole'], [], name)