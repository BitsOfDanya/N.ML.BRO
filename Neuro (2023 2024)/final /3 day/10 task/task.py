import numpy as np
import pandas as pd

# Загрузка тренировочных данных
train_data_path = 'data_real_train.csv'
train_data = pd.read_csv(train_data_path)

# Вывод первых строк и общей информации о данных
print(train_data.head(), train_data.info(), train_data.describe())
