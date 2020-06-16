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
    neg_file = r"negtive_drug_disease.txt"
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
    X, y = shengcheng()
    skf = KFold(n_splits=10, shuffle=True, random_state=1)
    auc_sub = []
    pr_sub = []
    auc_menche = []
    pr_menche = []
    auc_phe = []
    pr_phe = []
    for train_index, test_index in skf.split(X, y):
        X = np.array(X)
        y = np.array(y)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        predict_res2 = hh('Hadamard', file1, X_train, X_test, y_train)
        false_positive_rate1, true_positive_rate1, thresholds1 = roc_curve(y_test, predict_res2)
        auc1 = auc(false_positive_rate1, true_positive_rate1)
        pr = average_precision_score(y_test, predict_res2)
        auc_sub.append(auc1)
        pr_sub.append(pr)

        false_positive_rate5, true_positive_rate5, pre5, rec5, pr5 = s_ab(X_test, y_test, 'phenonet_drug_disease_distance.txt',
                                                                          flag=1)

        auc5 = auc(false_positive_rate5, true_positive_rate5)
        auc_menche.append(auc5)
        pr_menche.append(pr5)

        false_positive_rate6, true_positive_rate6, pre6, rec6, pr6 = s_ab(X_test, y_test,
                                                                          'disease_drug_proximity.txt', flag=0)
        auc6 = auc(false_positive_rate6, true_positive_rate6)
        auc_phe.append(auc6)
        pr_phe.append(pr6)
    ave_auc_sub = np.mean(auc_sub)
    ave_pr_sub = np.mean(pr_sub)

    ave_auc_menche = np.mean(auc_menche)
    ave_pr_menche = np.mean(pr_menche)

    ave_auc_phe = np.mean(auc_phe)
    ave_pr_phe = np.mean(pr_phe)

    print("SubNet2Vec_AUC")
    print(auc_sub)
    print("PhenoNet_AUC")
    print(auc_menche)
    print("Proximity_AUC")
    print(auc_phe)
    print("subNet2vec_PR")
    print(pr_sub)
    print("PhenoNet_PR")
    print(pr_menche)
    print("Menche_PR")
    print(pr_phe)




    print("SubNet2vec:", ave_auc_sub, ave_pr_sub, np.var(auc_sub), np.var(pr_sub))
    print("PhenoNet:", ave_auc_menche, ave_pr_menche, np.var(auc_menche), np.var(pr_menche))
    print("Proximity:", ave_auc_phe, ave_pr_phe, np.var(auc_phe), np.var(pr_phe))
    plt.show()
if __name__ == '__main__':
    main()

