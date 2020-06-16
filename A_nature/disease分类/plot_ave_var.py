# import numpy as np
# from matplotlib import pyplot as plt
# n = 2
# X = np.arange(n)+1
# print(X)
# sub = [0.971703052630693, 0.9630368036100967]
# sub_var=[4.70e-05, 7.85e-05]
# menche = [0.823923029159704, 0.8406980469085765]
# menche_var=[0.0006101109413101438, 0.0008419549385601692]
# phe=[0.8906017045360451, 0.8852342396477381]
# phe_var=[0.0005327642510046447, 0.0008045525800484679]
#
# figsize = 9, 18
# figure, ax = plt.subplots(figsize=figsize)
#
# # 设置坐标刻度值的大小以及刻度值的字体
# # plt.tick_params(axis=x,labelsize=28)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(28) for label in labels]
#
#
# plt.bar(X, menche, yerr= menche_var,alpha=0.9, width = 0.2, facecolor = 'lightskyblue', edgecolor = 'white', label='Menche', lw=5,error_kw = {'ecolor' : '0', 'capsize' :10,"lw":6})
# plt.bar(X+0.2, phe,yerr=phe_var, alpha=0.9, width = 0.2, facecolor = 'yellowgreen', edgecolor = 'white', label='PhenoNet', lw=5,error_kw = {'ecolor' : '0', 'capsize' :10,"lw":6})
# plt.bar(X+0.4,sub,yerr=sub_var, alpha=0.9, width = 0.2, facecolor = 'orange', edgecolor = 'white', label='SubNet2vec', lw=5,error_kw = {'ecolor' : '0', 'capsize' :10,"lw":6})
# plt.xticks(X+0.2,["AUROC","AUPRC"])
#
#
# y_list=[0.8,0.9,1.0]
# plt.ylim(0.8, 1.0)
# plt.yticks(y_list)
#
# font1 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 18,
#          }
#
# plt.legend(loc="upper right",prop=font1)
# plt.savefig("ave_var_disease_disease.jpg",dpi=300)
# plt.savefig("ave_var_disease_disease.eps",dpi=300)
# plt.show()

import numpy as np
from matplotlib import pyplot as plt
n = 2
X = np.arange(n)+1
print(X)
sub = [0.7931767939160497 ,0.8060593479750864]
sub_var=[0.004229325367499398 ,0.004831068479883757]
menche = [0.5554385991158597, 0.6111934696243758]
menche_var=[0.0017335001211230207, 0.0041986135323725865]
phe=[0.6693518229547623, 0.7022606988174492]
phe_var=[0.004603616236534778, 0.006105152516657293]
plt.figure(figsize=(9,18))
plt.bar(X, menche, yerr= menche_var,alpha=0.9, width = 0.2, facecolor = 'lightskyblue', edgecolor = 'white', label='PhenoNet', lw=1,error_kw = {'ecolor' : '0.2', 'capsize' :6})
plt.bar(X+0.2, phe,yerr=phe_var, alpha=0.9, width = 0.2, facecolor = 'yellowgreen', edgecolor = 'white', label='Proximity', lw=1,error_kw = {'ecolor' : '0.2', 'capsize' :6})
plt.bar(X+0.4,sub,yerr=sub_var, alpha=0.9, width = 0.2, facecolor = 'orange', edgecolor = 'white', label='SubNet2vec', lw=1,error_kw = {'ecolor' : '0.2', 'capsize' :6})
plt.xticks(X+0.2,["AUROC","AUPRC"])
y_list=[0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.ylim(0.4, 1.0)
plt.yticks(y_list)
plt.legend(loc="upper right")
plt.savefig("ave_var_drug_disease.jpg",dpi=300)
plt.savefig("ave_var_drug_disease.eps",dpi=300)
plt.show()
