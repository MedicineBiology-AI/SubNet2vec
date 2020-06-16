#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# from svm import *
# # from svmutil import *
"""
画疾病的ROC曲线
"""
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score,precision_recall_curve,average_precision_score
from sklearn.svm import SVC
import sklearn.preprocessing as prep
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import precision_recall_curve,precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold,KFold
def feature(oper, file1,pairs):
    d_vector = pd.read_csv(file1, header=None, encoding='gb2312', delimiter='\t', index_col=0)
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
    X_train, X_test, y_train, y_test = train_test_split(pos_feature, pos_label, test_size= .2, random_state=0)
    print(len(X_train))
    print(len(X_test))
    return X_train, y_train, X_test, y_test
def split(pos_sample, neg_sample):
    X_train, X_test, y_train, y_test = train_test_split(pos_sample, neg_sample, test_size=.2, random_state=0)
    return X_train, y_train, X_test, y_test
def sample():
    pos_file = 'pos1.txt'
    neg_file = 'neg1.txt'
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
    return pos_sample[:int(len(pos_sample)/1)], neg_sample[:int(len(neg_sample)/1)]
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
    pairs=[]
    for lines in open(file,"r"):
        line=lines.strip().split()
    with open(file, 'r') as f:


        pairs = [line.strip().split('\t')[:2] for line in f]
    prob = []
    print("sample")
    print(sample[:10])
    print("pairs")
    print(pairs[:10])

    print(sample[1])
    print(pairs[0])
    for pair in sample:

        break
    #     if pair in pairs:
    #         index = pairs.index(pair)
    #     else:
    #         index = pairs.index([pair[0], pair[1]])
    #     prob.append(value[index])
    #
    # false_positive_rate1, true_positive_rate1, thresholds1 = roc_curve(y_test, prob)
    # precision, recall, thresholds = precision_recall_curve(y_test, prob)
    #
    # pr = average_precision_score(y_test, prob)
    # return false_positive_rate1, true_positive_rate1, precision, recall, pr

def hh(opear,file1,X_train,X_test,y_train):
    train_feature1 = feature(opear, file1, X_train)
    test_feature1 = feature(opear, file1, X_test)
    train_feature1, test_feature1 = standard_scale(train_feature1, test_feature1)
    neigh = SVC(probability=True)
    neigh.fit(train_feature1, y_train)
    res1 = neigh.predict_proba(test_feature1)
    # res1=  neigh.predict(test_feature1)

    predict_res1 = list()
    for each in res1:
        predict_res1.append(each[1])
    return predict_res1
def main():
    auc_list=[]
    X, y = shengcheng()
    file1 = "result3/sub+gene_p_0.25.txt"
    ave_y_pre, had_y_pre, l1_y_pre, l2_y_pre, y_lab, Menche_X_test = [], [], [], [], [], []
    skf = KFold(n_splits=10, shuffle=True, random_state=1)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        Menche_X_test = Menche_X_test + list(X_test)
        y_lab = y_lab + list(y_test)

        # predict_res1 = hh('Average', file1, X_train, X_test, y_train)
        # ave_y_pre = ave_y_pre + predict_res1
        #
        # predict_res2 = hh('Hadamard', file1, X_train, X_test, y_train)
        # had_y_pre = had_y_pre + predict_res2
        #
        # predict_res3 = hh('L1', file1, X_train, X_test, y_train)
        # l1_y_pre = l1_y_pre + predict_res3
        #
        # predict_res4 = hh('L2', file1, X_train, X_test, y_train)
        # l2_y_pre = l2_y_pre + predict_res4
    # print("average")
    # p1 = precision_score(y_lab, ave_y_pre)
    # r1 = recall_score(y_lab, ave_y_pre)
    # f1 = f1_score(y_lab, ave_y_pre)
    # print(p1, r1, f1)
    #
    # print("Hadamard")
    # p2 = precision_score(y_lab, had_y_pre)
    # r2 = recall_score(y_lab, had_y_pre)
    # f2 = f1_score(y_lab, had_y_pre)
    # print(p2, r2, f2)
    #
    # print("L1")
    # p3 = precision_score(y_lab, l1_y_pre)
    # r3 = recall_score(y_lab, l1_y_pre)
    # f3 = f1_score(y_lab, l1_y_pre)
    # print(p3, r3, f3)
    #
    # print("L2")
    # p4 = precision_score(y_lab, l2_y_pre)
    # r4 = recall_score(y_lab, l2_y_pre)
    # f4 = f1_score(y_lab, l2_y_pre)
    # print(p4, r4, f4)
    # false_positive_rate1, true_positive_rate1, thresholds1 = roc_curve(y_lab, ave_y_pre)
    # auc1 = auc(false_positive_rate1, true_positive_rate1)
    # print(auc1)
    # auc_list.append(auc1)
    #
    # false_positive_rate2, true_positive_rate2, thresholds2 = roc_curve(y_lab, had_y_pre)
    # auc2 = auc(false_positive_rate2, true_positive_rate2)
    # print(auc2)
    # auc_list.append(auc2)
    #
    # false_positive_rate3, true_positive_rate3, thresholds3 = roc_curve(y_lab, l1_y_pre)
    # auc3 = auc(false_positive_rate3, true_positive_rate3)
    # print(auc3)
    # auc_list.append(auc3)
    #
    # false_positive_rate4, true_positive_rate4, thresholds4 = roc_curve(y_lab, l2_y_pre)
    # auc4 = auc(false_positive_rate4, true_positive_rate4)
    # print(auc4)
    # auc_list.append(auc4)


    false_positive_rate5, true_positive_rate5, pre5, rec5, pr5 = s_ab(Menche_X_test, y_lab, 's_ab1.txt', flag=0)
    auc5=auc(false_positive_rate5, true_positive_rate5)
    false_positive_rate6, true_positive_rate6, pre6, rec6, pr6 = s_ab(Menche_X_test, y_lab, 'phenonet_result1.txt', flag=1)
    auc6=auc(false_positive_rate6, true_positive_rate6)
    #

    # plt.plot(false_positive_rate5, true_positive_rate5, label='Menche (AUROC=%0.3f)' % auc5)
    #
    # plt.plot(false_positive_rate6, true_positive_rate6, label='PhenoNet (AUROC=%0.3f)' % auc6)
    #
    # plt.plot(false_positive_rate1, true_positive_rate1, label='Average' + '(AUROC=%0.3f)'
    #                                                           % auc1)
    # plt.plot(false_positive_rate2, true_positive_rate2, label='Hadamard' + '(AUROC=%0.3f)'
    #                                                           % auc2)
    # plt.plot(false_positive_rate3, true_positive_rate3, label='L1' + '(AUROC=%0.3f)'
    #                                                           % auc3)
    # plt.plot(false_positive_rate4, true_positive_rate4, label='L2' + '(AUROC=%0.3f)'
    #                                                           % auc4)
    # plt.xlim([0, 1.0])
    # plt.ylim([0, 1.0])
    # plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    # plt.title("ROC")
    # plt.legend(loc='lower right')
    # plt.ylabel('True Positive Rate', fontsize=12)
    # plt.xlabel('False Positive Rate', fontsize=12)
    # plt.savefig('drug_disease_ROC.eps', format='eps', dpi=300)
    # plt.savefig('drug_disease_ROC.jpg', format='jpg', dpi=300)
    #
    # plt.show()
    #

if __name__ == '__main__':
    main()
