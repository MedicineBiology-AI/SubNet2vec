from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.svm import SVC
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
false_positive_rate4, true_positive_rate4, thresholds4 = roc_curve([1,0,0,1], [0.5,0.3,0.6,0.1])
auc5=auc(false_positive_rate4,true_positive_rate4)
plt.figure()
plt.plot(false_positive_rate4, true_positive_rate4, label='Menche (AUROC=%0.3f)' % auc5)


plt.xlim([0, 1.0])
plt.ylim([0, 1.0])
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.title("ROC")
plt.legend(loc='lower right')
plt.ylabel('True Positive Rate', fontsize=12)
plt.xlabel('False Positive Rate', fontsize=12)
plt.savefig('1.eps', format='eps', dpi=300)
plt.savefig('1.jpg', format='jpg', dpi=300)

pre2, rec2, thresholds2 = precision_recall_curve([1,0,0,1],[0.5,0.3,0.6,0.1])
pr2 = average_precision_score([1,0,0,1], [0.5,0.3,0.6,0.1])


plt.figure()



plt.plot(rec2, pre2, label='Menche (AURR=%0.3f)' % pr2)
plt.xlim([0, 1.0])
plt.ylim([0, 1.0])

plt.plot([0, 1], [1, 0], color='black', linestyle='--')
plt.title("P-R")
plt.legend(loc='lower left')
plt.ylabel('Precision', fontsize=12)
plt.xlabel('Recall', fontsize=12)
plt.savefig('2.eps', format='eps', dpi=300)
plt.savefig('2.jpg', format='jpg', dpi=300)