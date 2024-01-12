import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pandas as pd

train_data_path = 'data_real_train.csv'
test_data_path = 'data_real_test.csv'

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

train_data_head = train_data.head()
test_data_head = test_data.head()

train_data_head, test_data_head

train_data = pd.read_csv(train_data_path, delimiter=';', index_col=False)
test_data = pd.read_csv(test_data_path, delimiter=';', index_col=False)

train_data_head = train_data.head()
test_data_head = test_data.head()

class_distribution = train_data['class'].value_counts(normalize=True)

print(train_data_head, test_data_head, class_distribution)
