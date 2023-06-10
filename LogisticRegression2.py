def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score, f1_score, auc
import matplotlib.pyplot as plt

import numpy as np
import os

os.chdir('/Users/alexanderjambor/Desktop/UCSD/SP23/BENG203/GroupProject/BRCAClassifier')

# 629 vs 690 features

X = np.loadtxt('./data/processed/1/normal_vs_cancer/X_filtered.csv', delimiter=',')

X = X[:, :400]


y = np.loadtxt('./data/processed/1/normal_vs_cancer/y_filtered.csv', delimiter=',')

X = np.log2(X+1)

X_train, y_train = X, y

X_test = np.loadtxt('./data/processed/2/normal_vs_cancer/X_filtered.csv', delimiter=',')
y_test = np.loadtxt('./data/processed/2/normal_vs_cancer/y_filtered.csv', delimiter=',')

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test[:, :400])

model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.5)
model.fit(X_train, y_train)


y_proba = model.predict_proba(X_test[:, :400])[:, 1]
y_pred = model.predict(X_test[:, :400])

fpr, tpr, _ = roc_curve(y_test, y_proba)
auroc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_test, y_proba)
auprc = auc(recall, precision)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

