import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

train_data = pd.read_csv('data_simple_train.csv')
train_data_split = train_data.iloc[:, 0].str.split(';', expand=True)
train_features = train_data_split.iloc[:, :-1].astype(float)
train_labels = train_data_split.iloc[:, -1].astype(int)

test_data = pd.read_csv('data_simple_test.csv')
test_data_split = test_data.iloc[:, 0].str.split(';', expand=True).astype(float)

X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

estimators = [
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
    ('catboost', CatBoostClassifier(verbose=0, random_state=42)),
    ('svc', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)))
]

final_estimator = LogisticRegression(max_iter=1000, random_state=42)

stacking_classifier = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=5
)

print("Обучение модели...")
stacking_classifier.fit(X_train, y_train)
print("Модель обучена.")

y_pred_val = stacking_classifier.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred_val)
print(f"Точность на валидационной выборке: {val_accuracy}")

cross_val_accuracy = cross_val_score(stacking_classifier, X_train, y_train, cv=5, scoring='accuracy').mean()
print(f"Средняя точность кросс-валидации: {cross_val_accuracy}")

test_predictions = stacking_classifier.predict(test_data_split)

predictions_df = pd.DataFrame(test_predictions, columns=['class'])
predictions_df.to_csv('predictions_final_improved.csv', index=False, header=True)

final_score = 6 * abs(1 - (1-val_accuracy)*2)
print(f"Итоговая оценка: {final_score}")
