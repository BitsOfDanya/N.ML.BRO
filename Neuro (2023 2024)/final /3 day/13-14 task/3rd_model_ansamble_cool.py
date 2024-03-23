import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

train_data = pd.read_csv('data_real_train.csv')
train_data_split = train_data.iloc[:, 0].str.split(';', expand=True)
train_features = train_data_split.iloc[:, :-1].astype(float)
train_labels = train_data_split.iloc[:, -1].astype(int)

test_data = pd.read_csv('data_real_test.csv')
test_data_split = test_data.iloc[:, 0].str.split(';', expand=True).astype(float)

pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])
train_features_scaled = pipeline.fit_transform(train_features)
test_features_scaled = pipeline.transform(test_data_split)

smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(train_features_scaled, train_labels)
X_train, X_val, y_train, y_val = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
test_features_scaled_pca = pca.transform(test_features_scaled)

estimators = [
    ('rf', RandomForestClassifier(n_estimators=200)),
    ('et', ExtraTreesClassifier(n_estimators=200)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100)),
    ('catboost', CatBoostClassifier(learning_rate=0.01, n_estimators=100, verbose=0)),
    ('gb', GradientBoostingClassifier(n_estimators=100)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('lr', LogisticRegression(max_iter=1000)),
    ('svc', SVC(probability=True, kernel='linear')),
    ('gnb', GaussianNB()),
    ('dt', DecisionTreeClassifier(max_depth=5))
]

voting_classifier = VotingClassifier(estimators=estimators, voting='soft')

param_distributions = {
    'rf__n_estimators': [100, 300],
    'rf__max_depth': [5, 15, None],
    'et__n_estimators': [100, 300],
    'et__max_depth': [5, 15, None],
    'xgb__n_estimators': [50, 150],
    'xgb__learning_rate': [0.01, 0.1],
    'catboost__n_estimators': [50, 150],
    'catboost__learning_rate': [0.01, 0.1],
    'gb__n_estimators': [50, 150],
    'gb__learning_rate': [0.01, 0.1]
}

search = RandomizedSearchCV(voting_classifier, param_distributions, n_iter=30, scoring='accuracy', n_jobs=-1, cv=StratifiedKFold(5), verbose=3)
search.fit(X_train_pca, y_train)

best_model = search.best_estimator_

y_pred_val = best_model.predict(X_val_pca)
val_accuracy = accuracy_score(y_val, y_pred_val)

test_predictions = best_model.predict(test_features_scaled_pca)

predictions_str = ''.join(map(str, test_predictions))
predictions_file_path = 'final_predictions_ensemble_3rd_cool.csv'
with open(predictions_file_path, 'w') as f:
    f.write(predictions_str)

print(f"Точность на валидационной выборке: {val_accuracy}")
print(f"Итоговая оценка: {5 * (val_accuracy ** 2)}")
