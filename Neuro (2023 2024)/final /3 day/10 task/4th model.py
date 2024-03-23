import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

train_data = pd.read_csv('data_real_train.csv')
train_data_split = train_data.iloc[:, 0].str.split(';', expand=True)
train_features = train_data_split.iloc[:, :-1].astype(float)
train_labels = train_data_split.iloc[:, -1].astype(int)

test_data = pd.read_csv('data_real_test.csv')
test_data_split = test_data.iloc[:, 0].str.split(';', expand=True).astype(float)

pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])

train_features_scaled = pipeline.fit_transform(train_features)
test_features_scaled = pipeline.transform(test_data_split)

X_train, X_val, y_train, y_val = train_test_split(train_features_scaled, train_labels, test_size=0.2, random_state=42)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
catboost = CatBoostClassifier(verbose=0)
rf = RandomForestClassifier()

voting_classifier = VotingClassifier(estimators=[('xgb', xgb), ('catboost', catboost), ('rf', rf)], voting='soft')

params = {
    'xgb__n_estimators': [100, 200, 300],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [5, 10, 15],
    'catboost__iterations': [100, 200, 300],
    'catboost__learning_rate': [0.01, 0.05, 0.1]
}

search = RandomizedSearchCV(voting_classifier, param_distributions=params, n_iter=10, scoring='accuracy', n_jobs=-1, cv=5, verbose=3)
search.fit(X_train, y_train)

best_model = search.best_estimator_

y_pred_val = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred_val)

test_predictions = best_model.predict(test_features_scaled)

predictions_str = ''.join(map(str, test_predictions))

predictions_file_path = 'predictions_final_improved_4th.csv'
with open(predictions_file_path, 'w') as f:
    f.write(predictions_str)

print(f"Точность на валидационной выборке: {val_accuracy}")
print(f"Итоговая оценка: {5 * (val_accuracy ** 2)}")
