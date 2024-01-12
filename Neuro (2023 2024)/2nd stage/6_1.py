import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import numpy as np

train_data = pd.read_csv('data_real_train.csv', delimiter=';')
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1].astype('int')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model1 = XGBClassifier()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = LogisticRegression()

ensemble = VotingClassifier(estimators=[
    ('xgb', model1),
    ('rf', model2),
    ('gb', model3),
    ('lr', model4)
], voting='soft')

parameters = {
    'xgb__n_estimators': [50, 100, 200],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__max_depth': [3, 5, 7],
    'xgb__subsample': [0.6, 0.8, 1.0],
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [3, 5, 7],
    'gb__n_estimators': [50, 100, 200],
    'gb__learning_rate': [0.01, 0.05, 0.1],
    'gb__max_depth': [3, 5, 7]
}

random_search = RandomizedSearchCV(ensemble, parameters, n_iter=10, scoring='accuracy', cv=3, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f'Лучшие параметры: {random_search.best_params_}')
print(f'Улучшенная точность: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

test_data = pd.read_csv('data_real_test.csv', delimiter=';')
test_data_scaled = scaler.transform(test_data)
predictions = best_model.predict(test_data_scaled)
print(f'Predictions: {predictions}')
