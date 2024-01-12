import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

df = pd.read_csv('')

X = df.drop('stim', axis=1)
y = df['stim']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

clf = RandomForestClassifier(n_estimators=1000)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

test = pd.read_csv('EKG_test_data.csv')

test['stim'] = clf.predict(test)

res = test['stim']

test.to_csv('res.csv')

table = open("res.csv", "r").read().replace("\n", ",").split(",")
fop = open("res1.txt", "w")
a = []

for i in range(180001+180002, len(table), 180002):
    a.append(table[i])

fop.write(" ".join(a))
fop.close()
