import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

def load_and_prepare_data(file_path, has_labels=True):
    data = pd.read_csv(file_path, header=None, sep=';')
    if has_labels:
        data = data[data.iloc[:, -1] != 'class']
        features_df = data.iloc[:, :-1].astype(float)
        labels_series = data.iloc[:, -1].astype(int)
        return features_df, labels_series
    else:
        return data.astype(float)

train_data_path = 'data_simple_train.csv'
train_features, train_labels = load_and_prepare_data(train_data_path)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_features)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, train_labels, test_size=0.2, random_state=42)

X_train_expanded = np.expand_dims(X_train, axis=2)
X_val_expanded = np.expand_dims(X_val, axis=2)

# упрощенная модель... типо базы
model = Sequential([
    Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train_expanded.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

class_weights = {0: 1, 1: len(y_train) / (2 * np.bincount(y_train)[1])}

model.fit(X_train_expanded, y_train, validation_data=(X_val_expanded, y_val), epochs=500, batch_size=128, callbacks=[early_stopping, reduce_lr], class_weight=class_weights)

y_pred_val = model.predict(X_val_expanded).flatten()
y_pred_val = np.round(y_pred_val).astype(int)

accuracy = accuracy_score(y_val, y_pred_val)
conf_matrix = confusion_matrix(y_val, y_pred_val)
class_report = classification_report(y_val, y_pred_val)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

test_data_path = 'data_simple_test.csv'
test_features = load_and_prepare_data(test_data_path, has_labels=False)
test_features_scaled = scaler.transform(test_features)
test_features_expanded = np.expand_dims(test_features_scaled, axis=2)
test_predictions = model.predict(test_features_expanded).flatten()
test_predictions = np.round(test_predictions).astype(int)

test_predictions_str = ''.join(map(str, test_predictions))
print("Test Predictions:", test_predictions_str)
