import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

train_data = pd.read_csv('data_simple_train.csv')
train_data_split = train_data.iloc[:, 0].str.split(';', expand=True)
train_features = train_data_split.iloc[:, :-1].astype(float)
train_labels = train_data_split.iloc[:, -1].astype(int)

test_data = pd.read_csv('data_simple_test.csv')
test_data_split = test_data.iloc[:, 0].str.split(';', expand=True).astype(float)

X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

estimators = [
    ('rf', best_rf),
    ('svc', make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)))
]
final_estimator = LogisticRegression(random_state=42)
stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=5)
stacking_classifier.fit(X_train, y_train)

test_predictions = stacking_classifier.predict(test_data_split)

with open('predictions_improved.txt', 'w') as file:
    for prediction in test_predictions:
        file.write(str(prediction))
