import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV

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

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
test_features_scaled_pca = pca.transform(test_features_scaled)

estimator_for_feature_selection = RandomForestClassifier(n_estimators=100)
selector = RFECV(estimator_for_feature_selection, step=1, cv=StratifiedKFold(2), scoring='accuracy')

estimators = [
    ('rf', Pipeline(steps=[('feature_selection', selector), ('classifier', RandomForestClassifier(n_estimators=200))])),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100)),
    ('catboost', CatBoostClassifier(learning_rate=0.01, n_estimators=100, verbose=0)),
    ('gb', GradientBoostingClassifier(n_estimators=100))
]

voting_classifier = VotingClassifier(estimators=estimators, voting='soft')

param_distributions = {
    'rf__classifier__max_depth': [5, 10, None],
    'xgb__n_estimators': [50, 100],
    'xgb__learning_rate': [0.01, 0.1],
    'catboost__n_estimators': [50, 100],
    'catboost__learning_rate': [0.01, 0.1],
    'gb__n_estimators': [50, 100],
    'gb__learning_rate': [0.01, 0.1]
}

search = RandomizedSearchCV(voting_classifier, param_distributions, n_iter=10, scoring='accuracy', n_jobs=-1, cv=StratifiedKFold(5), verbose=3)
search.fit(X_train_pca, y_train)

best_model = search.best_estimator_

y_pred_val = best_model.predict(X_val_pca)
val_accuracy = accuracy_score(y_val, y_pred_val)

test_predictions = best_model.predict(test_features_scaled_pca)

predictions_str = ''.join(map(str, test_predictions))
predictions_file_path = 'predictions_final_ensemble_optimized_v2.csv'
with open(predictions_file_path, 'w') as f:
    f.write(predictions_str)

print(f"Точность на валидационной выборке: {val_accuracy}")
print(f"Итоговая оценка: {5 * (val_accuracy ** 2)}")
