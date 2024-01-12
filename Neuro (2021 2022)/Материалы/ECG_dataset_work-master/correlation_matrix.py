# -*- coding: utf-8 -*


import os
import numpy as np
import json
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

from raw_dataset_to_pandas_frame import (
    get_pandas_dataframe, get_n_most_freq_names
)

# строим пандас таблицу, столбец - заболевание, строка -пациент
folder_with_files = 'C:\\ecg'  # путь к папке с исходными файлами .edf, .json
df = get_pandas_dataframe(folder_with_files)

# выбираем топ - n болезней
list_of_diagnoses = get_n_most_freq_names(n=15, folder_with_files = folder_with_files)

# осталвляем в табдице только эти эн столюбцов
df = df[list_of_diagnoses]
#print(tabulate(df, headers='keys', tablefmt='psql'))
#print(tabulate(df.describe(), headers='keys', tablefmt='psql'))

# рисуем для нее матрицу коэфициентов кореляции пирсона между столбацми (болезнями)
plt.clf()
f, ax = plt.subplots(figsize=(18, 18))
corr = df.corr(method='pearson')
print(tabulate(corr, headers='keys', tablefmt='psql'))
sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, vmin=-1, vmax=1)
plt.show()


