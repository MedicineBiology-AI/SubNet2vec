#coding=utf-8
import matplotlib.pyplot as plt
y1=[0.23157751, 0.32594265, 0.37651398 ,0.40709573, 0.43624726, 0.46154052,
 0.48167952 ,0.50271972, 0.51820565, 0.5328876 , 0.54679141, 0.56127827,
 0.57151211, 0.58390258, 0.59319869, 0.60795248, 0.61713856, 0.62865186,
 0.63408591, 0.64562817]
y2=[0.20650631 ,0.31518724 ,0.37478045 ,0.415589 ,  0.4625665  ,0.50444482,
 0.53094214, 0.55798643, 0.5845152,  0.60419218, 0.63282007 ,0.65524649,
 0.67166447 ,0.68687641 ,0.69987227, 0.71074131, 0.72225325 ,0.7348287,
 0.74705685, 0.75407797]
y3=[0.17684385, 0.29009479, 0.37906402 ,0.44774129 ,0.49162526, 0.53810366,
 0.57467871 ,0.60775284 ,0.64719573 ,0.67959946 ,0.69324308 ,0.71406664,
 0.73254883 ,0.74657316, 0.76456894, 0.77919163 ,0.79552721, 0.80242468,
 0.81710522, 0.83393038]

x1=[str(i) for i in range(1,21)]

plt.plot(x1,y1,label='Menche',linewidth=3,color='orange',marker='^',markerfacecolor='g',markersize=8)
plt.plot(x1,y2,label='PhenoNet',linewidth=3,color='c',marker='*',markerfacecolor='g',markersize=8)
plt.plot(x1,y3,label='Subnet2vec',linewidth=3,color='r',marker='o',markerfacecolor='g',markersize=8)
plt.xlabel('K')
plt.ylabel('HR@K')
plt.legend()
plt.savefig("HR@K_disease_disease.jpg")
plt.savefig("HR@K_disease_disease.eps")
plt.show()
# #coding=utf-8
# import matplotlib.pyplot as plt
# y1=[0.07534624 ,0.13826326 ,0.15286214 ,0.18528967 ,0.20692319 ,0.21956221,
#  0.23963714, 0.24941605 ,0.25937811 ,0.2741129,  0.28875889 ,0.29900011,
#  0.32740852, 0.34719999 ,0.35638366, 0.37308144 ,0.39335355 ,0.40668688,
#  0.41681988, 0.41931988]
# y2=[0.08723124 ,0.11905568, 0.13461298 ,0.15778793, 0.16546505 ,0.17443228,
#  0.18665006, 0.20475321 ,0.21255541, 0.23263965,0.24014278 ,0.2476536,
#  0.25259865 ,0.25979958 ,0.26223861 ,0.2674767 , 0.27237466 ,0.2768545,
#  0.29253085, 0.30230976]
# y3=[0.0485236 , 0.1024114 , 0.13182786, 0.18282907, 0.22208186, 0.2661225,
#  0.29633827, 0.32915457, 0.36234486 ,0.3717103,  0.39232805, 0.4146039,
#  0.43358349 ,0.45508096 ,0.46998668, 0.48198956, 0.50247657 ,0.51749655,
#  0.5270405 , 0.54924143]
# x1=[str(i) for i in range(1,21)]
# plt.plot(x1,y1,label='PhenoNet',linewidth=3,color='orange',marker='^',markerfacecolor='g',markersize=8)
# plt.plot(x1,y2,label='Proximity',linewidth=3,color='c',marker='*',markerfacecolor='g',markersize=8)
# plt.plot(x1,y3,label='Subnet2vec',linewidth=3,color='r',marker='o',markerfacecolor='g',markersize=8)
# plt.xlabel('K')
# plt.ylabel('HR@K')
# plt.legend()
# plt.savefig("HR@K_drug_disease.jpg")
# plt.savefig("HR@K_drug_disease.eps")
# plt.show()
