
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score, f1_score, auc
import matplotlib.pyplot as plt

import numpy as np
import os

os.chdir('/Users/alexanderjambor/Desktop/UCSD/SP23/BENG203/GroupProject/BRCAClassifier')


# X = np.loadtxt('./data/processed/1/normal_vs_cancer/X_filtered.csv', delimiter=',')
# y = np.loadtxt('./data/processed/1/normal_vs_cancer/y_filtered.csv', delimiter=',')

X = np.loadtxt('./data/processed/1/recurrent_vs_nonrecurrent/X_filtered.csv', delimiter=',')
y = np.loadtxt('./data/processed/1/recurrent_vs_nonrecurrent/y_filtered.csv', delimiter=',')

X = np.log2(X+1)

est = LogisticRegression(class_weight='balanced', solver='saga', penalty='elasticnet', l1_ratio=0.5)

X_train, y_train = X, y
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

rfe = RFE(est, n_features_to_select=200)
rfe.fit(X_train, y_train)

selected_features = rfe.support_
# feature_ranking = rfe.ranking_

X_train = X_train[:, selected_features]


auroc_vals = []
auprc_vals = []
accuracy_scores = []
f1_scores = []

fpr_vecs = []
tpr_vecs = []
precision_vecs = []
recall_vecs = []

# model = LogisticRegression(class_weight='balanced', solver='saga', penalty='elasticnet', l1_ratio=0.5)

model = SVC(probability=True, kernel='linear', class_weight='balanced')

# model = RandomForestClassifier(oob_score=True)


cv = StratifiedKFold(n_splits=5)


for train_idx, test_idx in cv.split(X[:, selected_features], y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train = np.log2(X_train+1)
    X_test = np.log2(X_test+1)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # smote = SMOTE(random_state=42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)

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
plt.title('Precision-Recall Curve')

plt.subplot(1, 2, 2)
for idx, _ in enumerate(fpr_vecs):
    plt.plot(fpr_vecs[idx], tpr_vecs[idx])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')

plt.legend()
plt.show()

#################


X_train, y_train = X, y
X_train = np.log2(X_train+1)
X_train = X_train[:, selected_features]

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)


# X_test = np.loadtxt('./data/processed/2/normal_vs_cancer/X_filtered.csv', delimiter=',')
# y_test = np.loadtxt('./data/processed/2/normal_vs_cancer/y_filtered.csv', delimiter=',')

X_test = np.loadtxt('./data/processed/2/recurrent_vs_nonrecurrent/X_filtered.csv', delimiter=',')
y_test = np.loadtxt('./data/processed/2/recurrent_vs_nonrecurrent/y_filtered.csv', delimiter=',')


X_test = X_test[:, selected_features]
X_test = np.log2(X_test+1)
X_test = scaler.transform(X_test)

model = SVC(probability=True, kernel='linear', class_weight='balanced')
# model = RandomForestClassifier(oob_score=True)
# model = LogisticRegression(class_weight='balanced', solver='saga', penalty='elasticnet', l1_ratio=0.5)

model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

fpr, tpr, _ = roc_curve(y_test, y_proba)
auroc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_test, y_proba)
auprc = auc(recall, precision)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)



