def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score, f1_score, auc
import matplotlib.pyplot as plt

import numpy as np
import os

os.chdir('/Users/alexanderjambor/Desktop/UCSD/SP23/BENG203/GroupProject/BRCAClassifier')

# 629 vs 690 features

# X = np.loadtxt('./data/processed/normal_vs_cancer/X_filtered.csv', delimiter=',')
# y = np.loadtxt('./data/processed/normal_vs_cancer/y_filtered.csv', delimiter=',')

X = np.loadtxt('./data/processed/recurrent_vs_nonrecurrent/X_filtered.csv', delimiter=',')
y = np.loadtxt('./data/processed/recurrent_vs_nonrecurrent/y_filtered.csv', delimiter=',')


auroc_vals = []
auprc_vals = []
accuracy_scores = []
f1_scores = []

fpr_vecs = []
tpr_vecs = []
precision_vecs = []
recall_vecs = []
oob_errors = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

model = RandomForestClassifier(oob_score=True)

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    oob_error = 1 - model.oob_score_
    oob_errors.append(oob_error)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fpr_vecs.append(fpr)
    tpr_vecs.append(tpr)
    auroc = auc(fpr, tpr)
    auroc_vals += [auroc]

    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    precision_vecs.append(precision)
    recall_vecs.append(recall)
    auprc = auc(recall, precision)
    auprc_vals += [auprc]

    accuracy = accuracy_score(y_test, y_pred)
    print(y_test)
    print(y_pred)
    accuracy_scores.append(accuracy)
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
avg_f1 = sum(f1_scores) / len(f1_scores)
avg_auroc = sum(auroc_vals) / len(auroc_vals)
avg_auprc = sum(auprc_vals) / len(auprc_vals)

print("<Accuracy>:", avg_accuracy)
print("<F1 Score>:", avg_f1)
print("<AUROC>:", avg_auroc)
print("<AUPRC>:", avg_auprc)

plt.subplot(1, 2, 1)
for idx, _ in enumerate(recall_vecs):
    plt.plot(recall_vecs[idx], precision_vecs[idx])
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.subplot(1, 2, 2)
for idx, _ in enumerate(fpr_vecs):
    plt.plot(fpr_vecs[idx], tpr_vecs[idx])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()
plt.show()
