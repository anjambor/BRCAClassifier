def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegressionCV

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

l1_ratios = [x/10 for x in range(0, 11, 1)]
model_tmp = LogisticRegressionCV(solver='saga', cv=5, scoring='f1', class_weight='balanced', penalty='elasticnet', Cs=10, l1_ratios=l1_ratios)
model_tmp = model_tmp.fit(X, y)

best_C = model_tmp.C_.item()
best_l1_ratio = model_tmp.l1_ratio_.item()

model = LogisticRegression(solver='saga', class_weight='balanced', penalty='elasticnet', C=best_C, l1_ratio=best_l1_ratio)

auroc_vals = []
auprc_vals = []
accuracy_scores = []
f1_scores = []

fpr_vecs = []
tpr_vecs = []
precision_vecs = []
recall_vecs = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

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
