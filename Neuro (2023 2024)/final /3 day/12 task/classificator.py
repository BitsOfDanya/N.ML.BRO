import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data_path = "P300_dataSet.txt"
columns = [f"feature_{i}" for i in range(1, 11)]
columns[-1] = "target"
data = pd.read_csv(data_path, sep=" ", names=columns)

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(kernel="linear", random_state=42)

rf_scores = cross_val_score(rf_clf, X_train, y_train, cv=5)
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=5)

print("Random Forest Classifier Accuracy:", rf_scores.mean())
print("SVM Classifier Accuracy:", svm_scores.mean())

best_clf = rf_clf if rf_scores.mean() > svm_scores.mean() else svm_clf
best_clf.fit(X_train, y_train)

test_accuracy = accuracy_score(y_test, best_clf.predict(X_test))
print("Test Accuracy:", test_accuracy)

channel_with_max_p300 = X.mean().idxmax()
channel_data = data[channel_with_max_p300]
target_data = channel_data[data["target"] == 1]
non_target_data = channel_data[data["target"] == 0]

plt.figure(figsize=(10, 6))
plt.hist(target_data, bins=50, color="blue", alpha=0.5, label="Target")
plt.hist(non_target_data, bins=50, color="red", alpha=0.5, label="Non-Target")
plt.title("Distribution of P300 Potential for Channel {}".format(channel_with_max_p300))
plt.xlabel("P300 Potential")
plt.ylabel("Frequency")
plt.legend()
plt.show()
