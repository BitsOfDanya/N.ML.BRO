import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, plot_importance
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import matplotlib.pyplot as plt

train_data = pd.read_csv('data_real_train.csv', delimiter=';')
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1].astype('int')

# создание полиномиальных признаков
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Балансировка классов
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# для оптимизации
def objective(space):
    model = XGBClassifier(n_estimators=int(space['n_estimators']),
                          max_depth=int(space['max_depth']),
                          learning_rate=space['learning_rate'],
                          min_child_weight=space['min_child_weight'],
                          subsample=space['subsample'],
                          colsample_bytree=space['colsample_bytree'],
                          gamma=space['gamma'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return {'loss': -accuracy, 'status': STATUS_OK}

# для поиска
space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 50),
    'max_depth': hp.choice('max_depth', np.arange(3, 10, dtype=int)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'min_child_weight': hp.uniform('min_child_weight', 1, 6),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'gamma': hp.uniform('gamma', 0.0, 0.5)
}

trials = Trials()
best_hyperparams = fmin(fn=objective,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=100,
                        trials=trials)

print("Лучшие гиперпараметры:", best_hyperparams)

best_model = XGBClassifier(n_estimators=int(best_hyperparams['n_estimators']),
                           max_depth=int(best_hyperparams['max_depth']),
                           learning_rate=best_hyperparams['learning_rate'],
                           min_child_weight=best_hyperparams['min_child_weight'],
                           subsample=best_hyperparams['subsample'],
                           colsample_bytree=best_hyperparams['colsample_bytree'],
                           gamma=best_hyperparams['gamma'])
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("Улучшенная точность:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

plot_importance(best_model)
plt.show()

test_data = pd.read_csv('data_real_test.csv', delimiter=';')
test_data_poly = poly.transform(test_data)
test_data_scaled = scaler.transform(test_data_poly)
predictions = best_model.predict(test_data_scaled)
print(f'Predictions: {predictions}')
