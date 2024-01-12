import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer
from imblearn.over_sampling import SVMSMOTE
from imblearn.pipeline import Pipeline as IMBPipeline
import xgboost as xgb

def split_row(row):
    values = row.split(';')
    features = values[:-1]
    label = values[-1]
    return features, label

def load_and_prepare_data(file_path, has_labels=True):
    data = pd.read_csv(file_path)
    features = []
    labels = []
    for index, row in data.iterrows():
        if has_labels:
            feature, label = split_row(row.iloc[0])
            labels.append(label)
        else:
            feature = row.iloc[0].split(';')
        features.append(feature)
    features_df = pd.DataFrame(features).astype(float)
    if has_labels:
        labels_series = pd.Series(labels).astype(int)
        return features_df, labels_series
    else:
        return features_df

train_features, train_labels = load_and_prepare_data('data_simple_train.csv')
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
X_train, X_val, y_train, y_val = train_test_split(train_features_scaled, train_labels, test_size=0.2, random_state=42)

pipeline = IMBPipeline([
    ('sampling', SVMSMOTE()),
    ('classification', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

parameters = {
    'classification__max_depth': [3, 4, 5, 6, 7, 8],
    'classification__learning_rate': [0.01, 0.05, 0.1, 0.15],
    'classification__n_estimators': [50, 100, 150, 200],
    'classification__min_child_weight': [1, 2, 3],
    'classification__gamma': [0, 0.1, 0.2],
    'classification__subsample': [0.7, 0.8, 0.9],
    'classification__colsample_bytree': [0.7, 0.8, 0.9],
    'classification__lambda': [1, 1.5, 2],
    'classification__alpha': [0, 0.5, 1]
}

def custom_metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

scorer = make_scorer(custom_metric)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(pipeline, parameters, cv=cv, scoring=scorer, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Лучшие параметры:", grid_search.best_params_)

best_model = grid_search.best_estimator_['classification']
best_model.fit(X_train, y_train)

threshold = 0.35  # Новый порог
y_pred_proba = best_model.predict_proba(X_val)[:, 1]
y_pred_val = (y_pred_proba > threshold).astype(int)

accuracy = accuracy_score(y_val, y_pred_val)
conf_matrix = confusion_matrix(y_val, y_pred_val)
class_report = classification_report(y_val, y_pred_val)

print("Modified Threshold Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

test_features = load_and_prepare_data('data_simple_test.csv', has_labels=False)
test_features_scaled = scaler.transform(test_features)
test_predictions_proba = best_model.predict_proba(test_features_scaled)[:, 1]
test_predictions = (test_predictions_proba > threshold).astype(int)

test_predictions_str = ''.join(map(str, test_predictions))
print("Test Predictions:", test_predictions_str)
