def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score, f1_score, auc
import matplotlib.pyplot as plt


import numpy as np
import os

os.chdir('/Users/alexanderjambor/Desktop/UCSD/SP23/BENG203/GroupProject/BRCAClassifier')

# 629 vs 690 features

X = np.loadtxt('./data/processed/1/recurrent_vs_nonrecurrent/X_filtered.csv', delimiter=',')
y = np.loadtxt('./data/processed/1/recurrent_vs_nonrecurrent/y_filtered.csv', delimiter=',')

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=3)

model = SVC(probability=True, kernel='linear', class_weight='balanced')

imp_vecs = []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)

    perm_imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=23, scoring='f1')
    imps = perm_imp.importances_mean
    imp_vecs += [imps]


imp_mat = np.array(imp_vecs)
sum_imp_vec = np.sum(imp_mat, axis=0)
idx_imp = [(idx, imp) for idx, imp in enumerate(sum_imp_vec)]

idx_imp = sorted(idx_imp, reverse=True, key=lambda x: x[1])

last_zero_idx = [x[0] for x in idx_imp if x[1]==0][-1]
end_idx = [idx for idx, x in enumerate(idx_imp) if x[0]==last_zero_idx][0] + 1
keep_idxs = [x[0] for x in idx_imp[:int(end_idx/4)]]

##############

X_test = np.loadtxt('./data/processed/2/recurrent_vs_nonrecurrent/X_filtered.csv', delimiter=',')
y_test = np.loadtxt('./data/processed/2/recurrent_vs_nonrecurrent/y_filtered.csv', delimiter=',')

X_test = X_test[:, keep_idxs]

model = SVC(probability=True, kernel='linear', class_weight='balanced')

scaler = StandardScaler()
X_train = X[:, keep_idxs]
y_train = y

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

fpr, tpr, _ = roc_curve(y_test, y_proba)
auroc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_test, y_proba)
auprc = auc(recall, precision)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)



