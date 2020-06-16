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
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
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


def feature(oper, file1, pairs):
    # file_path = '.\\test_data\\test\\'"result3/"+
    d_vector = pd.read_csv("result3/" + file1, header=None, encoding='gb2312', delimiter='\t', index_col=0)
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
    # pos_file = '.\\test_data\\test\\pos.txt'
    # neg_file = '.\\test_data\\test\\neg.txt'
    pos_file = 'pos1.txt'
    neg_file = 'neg1.txt'
    # pos_file = 'MF_GE.txt'
    # neg_file = 'neg_MF_GE.txt'
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
    # print(pos_sample)
    return pos_sample[:int(len(pos_sample) / 1)], neg_sample[:int(len(neg_sample) / 1)]


def shengcheng():
    pos_sample, neg_sample = sample()
    pos_label = [1] * len(pos_sample)
    neg_label = [0] * len(neg_sample)
    pos_label.extend(neg_label)
    pos_sample.extend(neg_sample)
    X_train, y_train, X_test, y_test = split(pos_sample, pos_label)
    return X_train, y_train, X_test, y_test


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
            index = pairs.index([pair[1], pair[0]])
        prob.append(value[index])
    false_positive_rate1, true_positive_rate1, thresholds1 = roc_curve(y_test, prob)
    precision, recall, thresholds = precision_recall_curve(y_test, prob)
    pr = average_precision_score(y_test, prob)
    return false_positive_rate1, true_positive_rate1, precision, recall, pr


def main():
    oper_names = ['Hadamard']
    # 'Average','Hadamard', 'L1', 'L2'
    X_train, y_train, X_test, y_test = shengcheng()
    for file1 in os.listdir(r"E:\A_nature\code_pr1_pr2\result8"):
        print(file1)
        train_feature1 = feature('Hadamard', file1, X_train)
        test_feature1 = feature('Hadamard', file1, X_test)
        train_feature1, test_feature1 = standard_scale(train_feature1, test_feature1)
        neigh = SVC(probability=True)
        neigh.fit(train_feature1, y_train)
        res1 = neigh.predict_proba(test_feature1)
        predict_res1 = list()
        for each in res1:
            predict_res1.append(each[1])
        false_positive_rate1, true_positive_rate1, thresholds1 = roc_curve(y_test, predict_res1)
        auc1 = auc(false_positive_rate1, true_positive_rate1)
        print(auc1)

if __name__ == '__main__':
    main()

