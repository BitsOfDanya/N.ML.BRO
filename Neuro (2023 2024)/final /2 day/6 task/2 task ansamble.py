import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel

train_data = pd.read_csv('data_simple_train.csv')
train_data_split = train_data.iloc[:, 0].str.split(';', expand=True)
train_features = train_data_split.iloc[:, :-1].astype(float)
train_labels = train_data_split.iloc[:, -1].astype(int)

test_data = pd.read_csv('data_simple_test.csv')
test_data_split = test_data.iloc[:, 0].str.split(';', expand=True).astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_features)
test_scaled = scaler.transform(test_data_split)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, train_labels, test_size=0.2, random_state=42)

selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, random_state=42)).fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_val_selected = selector.transform(X_val)
test_selected = selector.transform(test_scaled)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_estimators=50, max_depth=3)
xgb_model.fit(X_train_selected, y_train)

y_pred_val = xgb_model.predict(X_val_selected)
val_accuracy = accuracy_score(y_val, y_pred_val)
print(f"Точность на валидационной выборке: {val_accuracy}")

test_predictions = xgb_model.predict(test_selected)

predictions_df = pd.DataFrame(test_predictions, columns=['class'])
predictions_df.to_csv('predictions_final_optimized.csv', index=False)
print("Предсказания сохранены в файл 'predictions_final_optimized.csv'.")

final_score = 6 * abs(1 - (1-val_accuracy)*2)
print(f"Итоговая оценка: {final_score}")