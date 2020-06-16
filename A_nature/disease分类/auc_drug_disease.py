#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# from svm import *
# # from svmutil import *
"""
画疾病的ROC曲线
"""
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.svm import SVC
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from random import shuffle

def feature(oper, file1, pairs):

    d_vector = pd.read_csv("drug_disease/" + file1, header=None, encoding='gb2312', delimiter='\t', index_col=0)
    pair_vector = []
    for pair in pairs:
        if oper == 'Average':
            if pair[0] == pair[1]:
                continue
            pair_vector.append((d_vector.loc[pair[0]] + d_vector.loc[pair[1]]) / 2.)
        if oper == 'Hadamard':
            if pair[0] == pair[1]:
                continue
            pair_vector.append(d_vector.loc[pair[0]] * d_vector.loc[pair[1]])
        if oper == 'L1':
            if pair[0] == pair[1]:
                continue
            pair_vector.append(abs(d_vector.loc[pair[0]] - d_vector.loc[pair[1]]))
        if oper == 'L2':
            if pair[0] == pair[1]:
                continue

            pair_vector.append(np.square(d_vector.loc[pair[0]] - d_vector.loc[pair[1]]))
    return pair_vector


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test


def generate(pos_feature, neg_feature):
    pos_label = [1] * len(pos_feature)
    neg_label = [0] * len(neg_feature)
    pos_feature.extend(neg_feature)
    pos_label.extend(neg_label)
    X_train, X_test, y_train, y_test = train_test_split(pos_feature, pos_label, test_size=.2, random_state=0)
    print(len(X_train))
    print(len(X_test))
    return X_train, y_train, X_test, y_test

def split(pos_sample, neg_sample):
    X_train, X_test, y_train, y_test = train_test_split(pos_sample, neg_sample, test_size=.2, random_state=0)
    return X_train, y_train, X_test, y_test

def sample():
    pos_file = 'postive_drug_disease.txt'
    neg_file = "negtive_drug_disease.txt"
    pos_sample, neg_sample = [], []
    with open(pos_file, 'r') as f:
        for line in f:
            line_data = line.strip().split('\t')
            if line_data[0] == line_data[1]:
                continue
            pos_sample.append(line_data)
    with open(neg_file, 'r') as f:
        for line in f:
            line_data = line.strip().split('\t')
            if line_data[0] == line_data[1]:
                continue
            neg_sample.append(line_data)

    return pos_sample[:int(len(pos_sample) / 1)], neg_sample[:int(len(neg_sample) / 1)]

def shengcheng():
    pos_sample, neg_sample = sample()
    pos_label = [1] * len(pos_sample)
    neg_label = [0] * len(neg_sample)
    pos_label.extend(neg_label)
    pos_sample.extend(neg_sample)
    pos_sample = np.array(pos_sample)
    pos_label = np.array(pos_label)
    return pos_sample, pos_label


def s_ab(sample, y_test, file, flag):
    with open(file, 'r') as f:
        if flag == 0:
            value = [-float(line.strip().split('\t')[2]) for line in f]
        else:
            value = [float(line.strip().split('\t')[2]) for line in f]
    with open(file, 'r') as f:
        pairs = [line.strip().split('\t')[:2] for line in f]
    prob = []
    for pair in sample:
        if pair in pairs:
            index = pairs.index(pair)
        else:
            index = pairs.index([pair[0], pair[1]])
        prob.append(value[index])
    false_positive_rate1, true_positive_rate1, thresholds1 = roc_curve(y_test, prob)
    precision, recall, thresholds = precision_recall_curve(y_test, prob)
    pr = average_precision_score(y_test, prob)
    return false_positive_rate1, true_positive_rate1, precision, recall, pr

def hh(opear,file1,X_train,X_test,y_train):
    train_feature1 = feature(opear, file1, X_train)
    test_feature1 = feature(opear, file1, X_test)
    train_feature1, test_feature1 = standard_scale(train_feature1, test_feature1)
    neigh = SVC(probability=True)
    neigh.fit(train_feature1, y_train)
    res1 = neigh.predict_proba(test_feature1)

    predict_res1 = list()
    for each in res1:
        predict_res1.append(each[1])
    return predict_res1
def mm(file1):
    X, y = shengcheng()
    ave_y_pre, had_y_pre, l1_y_pre, l2_y_pre, y_lab, Menche_X_test = [], [], [], [], [], []
    skf = KFold(n_splits=10, shuffle=True, random_state=1)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        Menche_X_test = Menche_X_test + list(X_test)
        y_lab = y_lab + list(y_test)

        predict_res1 = hh('Average', file1, X_train, X_test, y_train)
        ave_y_pre = ave_y_pre + predict_res1

        predict_res2 = hh('Hadamard', file1, X_train, X_test, y_train)
        had_y_pre = had_y_pre + predict_res2

        predict_res3 = hh('L1', file1, X_train, X_test, y_train)
        l1_y_pre = l1_y_pre + predict_res3

        predict_res4 = hh('L2', file1, X_train, X_test, y_train)
        l2_y_pre = l2_y_pre + predict_res4
    return y_lab,ave_y_pre,had_y_pre,l1_y_pre,l2_y_pre, Menche_X_test

def main():
    file1 = "sub+gene_disease.txt"
    file2 = 'gene_disease.txt'
    file3 = "sub_disease.txt"

    y_lab_sub_gene, ave_y_pre_sub_gene, had_y_pre_sub_gene, l1_y_pre_sub_gene, l2_y_pre_sub_gene, Menche_X_test = mm(
        file1)
    y_lab_gene, ave_y_pre_gene, had_y_pre_gene, l1_y_pre_gene, l2_y_pre_gene, Menche_X_test_gene = mm(file2)
    y_lab_sub, ave_y_pre_sub, had_y_pre_sub, l1_y_pre_sub, l2_y_pre_sub, Menche_X_test_sub = mm(file3)

    # 画ROC
    false_positive_rate1, true_positive_rate1, thresholds1 = roc_curve(y_lab_sub_gene, ave_y_pre_sub_gene)
    auc1 = auc(false_positive_rate1, true_positive_rate1)
    print("Sub+gene Average AUC:", auc1)

    false_positive_rate2, true_positive_rate2, thresholds2 = roc_curve(y_lab_sub_gene, had_y_pre_sub_gene)
    auc2 = auc(false_positive_rate2, true_positive_rate2)
    print("Sub+gene Hadmard AUC:", auc2)

    false_positive_rate3, true_positive_rate3, thresholds3 = roc_curve(y_lab_sub_gene, l1_y_pre_sub_gene)
    auc3 = auc(false_positive_rate3, true_positive_rate3)
    print("Sub+gene L1 AUC:", auc3)

    false_positive_rate4, true_positive_rate4, thresholds4 = roc_curve(y_lab_sub_gene, l2_y_pre_sub_gene)
    auc4 = auc(false_positive_rate4, true_positive_rate4)
    print("Sub+gene L2 AUC:", auc4)

    false_positive_rate7, true_positive_rate7, thresholds7 = roc_curve(y_lab_gene, ave_y_pre_gene)
    auc7 = auc(false_positive_rate7, true_positive_rate7)
    print("gene Average AUC:", auc7)

    false_positive_rate8, true_positive_rate8, thresholds8 = roc_curve(y_lab_gene, had_y_pre_gene)
    auc8 = auc(false_positive_rate8, true_positive_rate8)
    print("gene Hadmard AUC:", auc8)

    false_positive_rate9, true_positive_rate9, thresholds9 = roc_curve(y_lab_gene, l1_y_pre_gene)
    auc9 = auc(false_positive_rate9, true_positive_rate9)
    print("gene L1 AUC:", auc9)

    false_positive_rate10, true_positive_rate10, thresholds10 = roc_curve(y_lab_gene, l2_y_pre_gene)
    auc10 = auc(false_positive_rate10, true_positive_rate10)
    print("gene L2 AUC:", auc10)

    false_positive_rate11, true_positive_rate11, thresholds11 = roc_curve(y_lab_sub, ave_y_pre_sub)
    auc11 = auc(false_positive_rate11, true_positive_rate11)
    print("Sub Average AUC:", auc11)

    false_positive_rate12, true_positive_rate12, thresholds12 = roc_curve(y_lab_sub, had_y_pre_sub)
    auc12 = auc(false_positive_rate12, true_positive_rate12)
    print("Sub Hadmard AUC:", auc12)

    false_positive_rate13, true_positive_rate13, thresholds13 = roc_curve(y_lab_sub, l1_y_pre_sub)
    auc13 = auc(false_positive_rate13, true_positive_rate13)
    print("Sub L1 AUC:", auc13)

    false_positive_rate14, true_positive_rate14, thresholds14 = roc_curve(y_lab_sub, l2_y_pre_sub)
    auc14 = auc(false_positive_rate14, true_positive_rate14)
    print("Sub L2 AUC:", auc14)

    false_positive_rate5, true_positive_rate5, pre5, rec5, pr5 = s_ab(Menche_X_test, y_lab_sub_gene,'phenonet_drug_disease_distance.txt',flag=1)

    auc5 = auc(false_positive_rate5, true_positive_rate5)
    print("PhenoNet AUC:", auc5)

    false_positive_rate6, true_positive_rate6, pre6, rec6, pr6 = s_ab(Menche_X_test, y_lab_sub_gene,
                                                                       'disease_drug_proximity.txt', flag=0)
    auc6 = auc(false_positive_rate6, true_positive_rate6)
    print("Proximity AUC:", auc6)

    plt.plot(false_positive_rate5, true_positive_rate5, label='PhenoNet (AUROC=%0.3f)' % auc5)

    plt.plot(false_positive_rate6, true_positive_rate6, label='Proximity (AUROC=%0.3f)' % auc6, )

    # plt.plot(false_positive_rate8, true_positive_rate8, label='Node2vec' + '(AUROC=%0.3f)' % auc8)
    plt.plot(false_positive_rate2, true_positive_rate2, label='SubNet2vec' + '(AUROC=%0.3f)' % auc2, color="red")

    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.title("ROC",fontsize=20)
    plt.legend(loc='lower right',fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('drug_disease_ROC.eps', format='eps', dpi=300)
    plt.savefig('drug_disease_ROC.jpg', format='jpg', dpi=300)

    plt.show()
    """
    画P_R
    """
    pr1 = average_precision_score(y_lab_sub_gene, ave_y_pre_sub_gene)
    print("Sub+gene Average PR:", pr1)

    pre2, rec2, thresholds2 = precision_recall_curve(y_lab_sub_gene, had_y_pre_sub_gene)
    pr2 = average_precision_score(y_lab_sub_gene, had_y_pre_sub_gene)
    print("Sub+gene Hadamard PR:", pr2)

    pr3 = average_precision_score(y_lab_sub_gene, l1_y_pre_sub_gene)
    print("Sub+gene L1 PR:", pr3)

    pr4 = average_precision_score(y_lab_sub_gene, l2_y_pre_sub_gene)
    print("Sub+gene L2 PR", pr4)

    pr7 = average_precision_score(y_lab_gene, ave_y_pre_gene)
    print("Gene Average PR:", pr7)

    pre8,rec8,thresholds8=precision_recall_curve(y_lab_gene,had_y_pre_gene)
    pr8 = average_precision_score(y_lab_gene, had_y_pre_gene)
    print("Gene Hadamard PR:", pr8)

    pr9 = average_precision_score(y_lab_gene, l1_y_pre_gene)
    print("Gene L1 PR:", pr9)

    pr10 = average_precision_score(y_lab_gene, l2_y_pre_gene)
    print("Gene L2 PR", pr10)

    pr11 = average_precision_score(y_lab_sub, ave_y_pre_sub)
    print("Sub Average PR:", pr11)

    pr12 = average_precision_score(y_lab_sub, had_y_pre_sub)
    print("Sub Hadamard PR:", pr12)

    pr13 = average_precision_score(y_lab_sub, l1_y_pre_sub)
    print("Sub L1 PR:", pr13)

    pr14 = average_precision_score(y_lab_sub, l2_y_pre_sub)
    print("Sub PR", pr14)

    plt.plot(rec5, pre5, label='PhenoNet (AURR=%0.3f)' % pr5)
    plt.plot(rec6, pre6, label='Proximity (AUPR=%0.3f)' % pr6)
    # plt.plot(rec8, pre8, label='Node2vec (AUPR=%0.3f)' % pr8)
    plt.plot(rec2, pre2, label='SubNet2vec (AUPR=%0.3f)' % pr2, color="red")

    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.plot([0, 1], [1, 0], color='black', linestyle='--')
    plt.title("P-R",fontsize=20)
    plt.legend(loc='lower left',fontsize=14)
    plt.ylabel('Precision', fontsize=15)
    plt.xlabel('Recall', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('drug_disease_PR.eps', format='eps', dpi=300)
    plt.savefig('drug_disease_PR.jpg', format='jpg', dpi=300)
    plt.show()

if __name__ == '__main__':
      main()
