import pandas as pd
from sklearn.preprocessing import StandardScaler

def split_data(row):
    values = row.split(';')
    features = values[:-1]  # все значения, кроме последнего
    label = values[-1]  # последнее значение - метка класса
    return features, label

def process_data(data_path):
    data = pd.read_csv(data_path)
    features = []
    labels = []

    for index, row in data.iterrows():
        feature, label = split_data(row[0])
        features.append(feature)
        labels.append(label)

    features_df = pd.DataFrame(features).astype(float)
    labels_series = pd.Series(labels).astype(int)
    return features_df, labels_series

train_features, train_labels = process_data('data_simple_train.csv')

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)

train_features_scaled_df = pd.DataFrame(train_features_scaled, columns=train_features.columns)

print(train_features_scaled_df.head(), train_labels.head())


train_data_path = 'data_simple_train.csv'

train_data = pd.read_csv(train_data_path, header=None, sep=';')

features_df = train_data.iloc[:, :-1]
labels_series = train_data.iloc[:, -1]

class_distribution = labels_series.value_counts()
print(class_distribution)
