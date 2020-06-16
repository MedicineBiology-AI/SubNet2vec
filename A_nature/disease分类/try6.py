# Credit: Josh Hemann

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

n_groups = 2
#drug-disease
# sub = [ 0.7679980525802301, 0.7932819411114941]
# sub_var=[0.00194535848241615, 0.005121156974660025]
# menche = [0.5977593547942106, 0.6412872397900178]
# menche_var=[0.0025789098830007436, 0.0035785782436437187]
# phe=[0.6550149536320921, 0.6901970094320005]
# phe_var=[0.0018882243998543692, 0.00460997404754374]
#
# fig, ax = plt.subplots(figsize=(9,9))
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(22) for label in labels]
# list1=[0.5,1.0]
# list1=np.array(list1)
# index =list1
# bar_width = 0.1
# opacity = 0.5
# error_config = {'ecolor': '0.3','capsize' :8,"lw":4}
#
# rects2 = ax.bar(index, menche, bar_width,
#                 alpha=opacity, color='blue',
#                 yerr=menche_var, error_kw=error_config,
#                 label='PhenoNet')
#
# rects3 = ax.bar(index + bar_width, phe, bar_width,
#                 alpha=opacity, color='green',
#                 yerr=phe_var, error_kw=error_config,
#                 label='Proximity')
#
# rects1 = ax.bar(index + bar_width*2, sub, bar_width,
#                 alpha=opacity, color='red',
#                 yerr=sub_var, error_kw=error_config,
#                 label='SubNet2vec')
# plt.ylim(0.4,1)
# # ax.set_xlabel('Group')
# # ax.set_ylabel('Scores')
# # ax.set_title('Scores by group and gender')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(('AUROC', 'AUPRC'))
# font1 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 18,
#          }
# ax.legend(prop=font1)
#
# fig.tight_layout()
# plt.savefig("ave_var_drug_disease.jpg",dpi=300)
# plt.savefig("ave_var_drug_disease.eps",dpi=300)
# plt.show()

sub = [0.9717030301884367, 0.963029684901031]
sub_var=[4.673700117852435e-05, 7.85503225675871e-05]
menche = [0.823923029159704 ,0.8406980469085765]
menche_var=[0.0006101109413101438, 0.0008419549385601692]
phe=[0.8906017045360451, 0.8852342396477381]
phe_var=[0.0005327642510046447 ,0.0008045525800484679]

fig, ax = plt.subplots(figsize=(9,9))
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(22) for label in labels]
list1=[0.5,1.0]
list1=np.array(list1)
index =list1
bar_width = 0.1
opacity = 0.5
error_config = {'ecolor': '0.3','capsize' :8,"lw":4}

rects2 = ax.bar(index, menche, bar_width,
                alpha=opacity, color='blue',
                yerr=menche_var, error_kw=error_config,
                label='Menche')

rects3 = ax.bar(index + bar_width, phe, bar_width,
                alpha=opacity, color='green',
                yerr=phe_var, error_kw=error_config,
                label='PhenoNet')

rects1 = ax.bar(index + bar_width*2, sub, bar_width,
                alpha=opacity, color='red',
                yerr=sub_var, error_kw=error_config,
                label='SubNet2vec')
plt.ylim(0.82,1)
# ax.set_xlabel('Group')
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('AUROC', 'AUPRC'))
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }
ax.legend(prop=font1)

fig.tight_layout()
plt.savefig("ave_var_disease_disease.jpg",dpi=300)
plt.savefig("ave_var_disease_disease.eps",dpi=300)
plt.show()