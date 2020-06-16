#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
    d_vector = pd.read_csv( file1, header=None, encoding='gb2312', delimiter='\t', index_col=0)
    pair_vector = []
    for pair in pairs:
        if oper == 'Average':
            pair_vector.append((d_vector.loc[pair[0]] + d_vector.loc[pair[1]]) / 2.)
        if oper == 'Hadamard':
            if pair[0] == pair[1]:
                continue
            pair_vector.append(d_vector.loc[pair[0]] * d_vector.loc[pair[1]])
        if oper == 'L1':
            pair_vector.append(abs(d_vector.loc[pair[0]] - d_vector.loc[pair[1]]))
        if oper == 'L2':
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
    pos_file = 'pos2.txt'
    neg_file = 'neg1.txt'
    pos_sample, neg_sample = [], []
    with open(pos_file, 'r') as f:
        for line in f:
            line_data = line.strip().split('\t')
            if line_data[0] == line_data[1]:
                continue
            pos_sample.append([line_data[0],line_data[1]])
    with open(neg_file, 'r') as f:
        for line in f:
            line_data = line.strip().split('\t')
            if line_data[0] == line_data[1]:
                continue
            neg_sample.append([line_data[0],line_data[1]])

    return pos_sample,neg_sample
    # return pos_sample[:int(len(pos_sample) / 1)], neg_sample[:int(len(neg_sample) / 1)]

def shengcheng():
    pos_sample, neg_sample = sample()
    pos_label = [1] * len(pos_sample)
    neg_label = [0] * len(neg_sample)
    pos_label.extend(neg_label)
    pos_sample.extend(neg_sample)
    pos_sample1 = np.array(pos_sample)
    pos_label1 = np.array(pos_label)
    return pos_sample1, pos_label1


def s_ab(sample, y_test, file, flag):
    sample1=[]
    for s in sample:
        p=[]
        p.append(s[0])
        p.append(s[1])
        sample1.append(p)
    value=[]
    with open(file, 'r') as f:
        for line in f:
            l = line.strip().split("\t")
            if flag == 0:
                    value.append(-float(l[2]))
                    value.append(-float(l[2]))
            else:
                    value.append(float(l[2]))
                    value.append(float(l[2]))
    pairs=[]
    for lines in open(file, 'r'):
        line=lines.strip().split("\t")
        p=[line[0],line[1]]
        pairs.append(p)
        p1=[line[1],line[0]]
        pairs.append(p1)
    prob = []
    for pair in sample1:
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
def predict(test,predict_res1):
    j = 0
    ave = []
    for i in range(len(test)):
        sum = [0 for _ in range(20)]
        pro = predict_res1[i * 100]
        j = j + 1
        list1 = predict_res1[i * 100:j * 100]
        list1.sort(reverse=True)
        for m in range(1, 21):
            if pro > list1[m]:  # TOP10
                sum[m - 1] = sum[m - 1] + 1
        ave.append(sum)
    # 维度为测试集中疾病的个数*20
    ave = np.array(ave)
    top = ave.mean(axis=0)
    return top

def main():
    """
    对于测试集中的一个正样本，采样99个负样本。输入到模型中，计算TOP(1-10)
    现在有801个疾病，801*801个关系，除去所有的正样本和训练集中的负样本。对测试集中每一个正样本，随机选取99个负样本，计算score
    1.所有的正样本
    2.去掉训练集
    3.所有的正样本-训练集中的正样本，就是测试集中的正样本

    """
    file6 = open("file1.txt", "a")
    file1 = "result3/sub+gene_p_0.25.txt"
    file2= "result3/gene_q_0.25.txt"
    #得到所有的正样本
    all_postive=set()
    for lines in open("pos2.txt"):
        line=lines.strip().split("\t")
        all_postive.add((line[0],line[1]))
    # print(len(all_postive))#1292个正样本

    #得到801个疾病的所有的数据
    all_disease=set()
    for lines in open("s_ab1.txt"):
        line=lines.strip().split("\t")
        all_disease.add(line[0])
        all_disease.add(line[1])
    # print(len(all_disease))#801疾病
    all_disease_pairs=set()
    for d1 in all_disease:
        for d2 in all_disease:
            if d1==d2:
                continue
            else:
                all_disease_pairs.add((d1,d2))
    # print(len(all_disease_pairs))

    dict_disease = {}
    for lines in open("phenonet_result1.txt"):
        line = lines.strip().split("\t")
        dict_disease[(line[0], line[1])] = float(line[2])
        dict_disease[(line[1], line[0])] = float(line[2])
    print("已知的有多少对疾病之间的关系:", len(dict_disease))
    dict_disease1 = {}
    for lines in open("s_ab1.txt"):
        line = lines.strip().split("\t")
        dict_disease1[(line[0], line[1])] = -float(line[2])
        dict_disease1[(line[1], line[0])] = -float(line[2])
    print("已知的有多少对疾病之间的关系:", len(dict_disease1))


    sub_ht_ave,node_ht_ave,menche_ht_ave,phe_ht_ave=[],[],[],[]
    X, y = shengcheng()
    skf = KFold(n_splits=10, shuffle=True, random_state=1)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        x_train_set=set()
        for i in X_train:
            x_train_set.add((i[0],i[1]))
        # print(len(x_train_set))#2280
        test=all_postive-x_train_set
        choise_disease_pairs=all_disease_pairs-all_postive-x_train_set
        # print(len(choise_disease_pairs))

        dict_choose_disease={}
        for i in choise_disease_pairs:
            if i[0] not in dict_choose_disease:
                dict_choose_disease[i[0]]=[]
            dict_choose_disease[i[0]].append((i[0],i[1]))
        predict_sample=[]
        for t in test:
            predict_sample.append((t[0], t[1]))
            shuffle(dict_choose_disease[t[0]])
            predict_sample=predict_sample+dict_choose_disease[t[0]][:99]

        prob,prob1 = [],[]
        for pair in predict_sample:
            prob.append(dict_disease[pair])
            prob1.append(dict_disease1[pair])

        y_train, y_test = y[train_index], y[test_index]
        predict_sample = np.array(predict_sample)
        train_feature1 = feature('Hadamard', file1, X_train)
        predict_feature1 = feature('Hadamard', file1, predict_sample)
        train_feature1, predict_feature1 = standard_scale(train_feature1, predict_feature1)

        train_feature2 = feature('Hadamard', file2, X_train)
        predict_feature2 = feature('Hadamard', file2, predict_sample)
        train_feature2, predict_feature2 = standard_scale(train_feature2, predict_feature2)

        neigh = SVC(probability=True)

        neigh.fit(train_feature1, y_train)
        res1 = neigh.predict_proba(predict_feature1)
        predict_res1 = list()
        for each in res1:
            predict_res1.append(each[1])

        top1=predict(test,predict_res1)

        sub_ht_ave.append(top1)

        neigh.fit(train_feature2, y_train)
        res2 = neigh.predict_proba(predict_feature2)
        predict_res2 = list()
        for each in res2:
            predict_res2.append(each[1])

        top4 = predict(test, predict_res2)
        node_ht_ave.append(top4)

        top2=predict(test,prob)
        phe_ht_ave.append(top2)

        top3=predict(test,prob1)
        menche_ht_ave.append(top3)
        predict_res1=np.array(predict_res1)

        print(predict_sample.shape)
        print(predict_res1.shape)
        pp=[]
        for p in predict_res1:
            pp.append([p])
        pp=np.array(pp)
        list3=np.hstack((predict_sample[1:], pp[1:]))
        for l in list3:
            file6.write(str(l[0]))
            file6.write("\t")
            file6.write(str(l[1]))
            file6.write("\t")
            file6.write(str(l[2]))
            file6.write("\n")

    sub_ht_ave=np.array(sub_ht_ave)
    node_ht_ave=np.array(node_ht_ave)
    menche_ht_ave=np.array(menche_ht_ave)
    phe_ht_ave=np.array(phe_ht_ave)
    print("应该是10*20")
    print(sub_ht_ave.shape,menche_ht_ave.shape,phe_ht_ave.shape)
    print("menche","phenoNet","node2vec","subNet2vec")
    menche_ht=menche_ht_ave.mean(axis=0)
    phe_ht=phe_ht_ave.mean(axis=0)
    node_ht=node_ht_ave.mean(axis=0)
    sub_ht=sub_ht_ave.mean(axis=0)
    file6.close()
    x1 = [str(i) for i in range(1, 21)]
    plt.plot(x1, menche_ht, label='Menche', linewidth=3, color='orange', marker='^', markerfacecolor='g', markersize=8)
    plt.plot(x1, phe_ht, label='PhenoNet', linewidth=3, color='c', marker='*', markerfacecolor='g', markersize=8)
    # plt.plot(x1, node_ht, label='Node2vec', linewidth=3, color='royalblue', marker='>', markerfacecolor='g', markersize=8)
    plt.plot(x1, sub_ht, label='Subnet2vec', linewidth=3, color='r', marker='o', markerfacecolor='g', markersize=8)
    plt.xlabel('K',fontsize=15)
    plt.ylabel('HR@K',fontsize=15)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=15)
    plt.savefig("HR@K_disease_disease.jpg")
    plt.savefig("HR@K_disease_disease.eps")
    plt.show()

if __name__ == '__main__':
    main()

