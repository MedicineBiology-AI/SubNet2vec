file1=open("disease_drug_quchong1.txt","w")
dict_dise_drug={}
for lines in open("file22.txt"):
    line=lines.strip().split("\t")
    dict_dise_drug[(line[0],line[1])]=float(line[2])
for key in dict_dise_drug:
    file1.write(key[0])
    file1.write("\t")
    file1.write(key[1])
    file1.write("\t")
    file1.write(str(dict_dise_drug[key]))
    file1.write("\n")
file1.close()


