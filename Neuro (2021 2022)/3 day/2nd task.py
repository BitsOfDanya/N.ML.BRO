# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

df = pd.read_csv('EDA_train.csv')

X = df.drop('target', axis = 1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

test = pd.read_csv('EDA_test.csv')

test['target'] = clf.predict(test)

res = test['target']

test.to_csv('res.csv')

table = open("res.csv", "r").read().replace("\n", ",").split(",")
fop = open("res1.txt", "w")
a = []

for i in range(180001+180002, len(table), 180002):
    a.append(table[i])

fop.write(" ".join(a))
fop.close()

print(dataset.head(20))
